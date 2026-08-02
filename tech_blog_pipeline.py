#!/usr/bin/env python3
"""Turn a technical talk and its slides into an illustrated, evidence-grounded blog.

The pipeline is deliberately staged: ingest -> slide vision -> evidence ledger ->
outline -> parallel section drafts -> global edit -> independent review -> repair.
Every expensive stage is cached, so an interrupted run can be resumed safely.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import html
from html.parser import HTMLParser
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import requests
from PIL import Image

from blog_docx import audit_docx, markdown_to_docx


API_BASE = "https://openrouter.ai/api/v1"
SCHEMA_VERSION = 1
PIPELINE_VERSION = "2026-08-02.1"
DEFAULT_ANALYST = "google/gemini-3.1-pro-preview"
DEFAULT_WRITER = "anthropic/claude-opus-4.6"
DEFAULT_CRITIC = "google/gemini-3.1-pro-preview"
DEFAULT_STT = "openai/gpt-4o-mini-transcribe"

STYLE_PROFILE = """采用图解型工程技术导读的一般写法，但绝不模仿任何特定作者的独特措辞。
- 从真实工程矛盾切入；先给系统全景，再沿真实依赖与因果关系逐层展开。
- 每节形成闭环：问题 -> 图/表 -> 图中元素 -> 因果机制 -> 参数/公式/数据锚点 -> 最小例子 -> 边界和小结。
- 从简单情形扩展到并行、多卡或边界情形，每次说明新增变量改变了什么。
- 图必须承担信息，正文要解释图中的箭头、编号、颜色或数据，不能只把图当装饰。
- 术语首次出现给中英文和一句定义；短段落；结论明确；不写空泛赞美。
- 明确区分材料事实、演讲者观点、合理推断和外部补充；没有证据就不补数字。
"""

REFERENCE_SAFETY = """参考文章只用于抽取抽象组织原则，不作为正文语料或技术事实来源。
不得复用参考文的标题模板、开场白、收尾句、比喻、例子、段落或配图；不得暗示自己是原作者。
除通用技术术语、API 名和代码符号外，不得出现与参考文连续 8 个及以上相同汉字。
成稿图片只能来自用户 PPT/PDF、视频、经核验的一手资料，或基于当前材料原创重绘。
"""


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp") as handle:
        handle.write(value)
        temp = Path(handle.name)
    temp.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2))


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        head = handle.read(4 * 1024 * 1024)
        digest.update(head)
        if stat.st_size > len(head):
            handle.seek(max(0, stat.st_size - 4 * 1024 * 1024))
            digest.update(handle.read(4 * 1024 * 1024))
    return {"path": str(path.resolve()), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sample_sha256": digest.hexdigest()}


def run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"命令执行失败：{command[0]}")
    return result.stdout.strip()


def timestamp(seconds: float) -> str:
    value = max(0, round(seconds))
    hours, rest = divmod(value, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def strip_fence(text: str) -> str:
    value = text.strip()
    match = re.fullmatch(r"```(?:markdown|md)?\s*(.*?)\s*```", value, re.I | re.S)
    return match.group(1).strip() if match else value


def parse_json_text(text: str) -> Any:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.I | re.S)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}|\[.*\]", value, re.S)
        if not match:
            raise RuntimeError("模型没有返回可解析的 JSON")
        return json.loads(match.group(0))


class OpenRouter:
    def __init__(self, api_key: str, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "Evidence Grounded Technical Blog Pipeline",
        }

    def _post(self, path: str, payload: dict[str, Any], timeout: int = 600) -> dict[str, Any]:
        error = "unknown error"
        for attempt in range(5):
            try:
                response = requests.post(API_BASE + path, headers=self.headers, json=payload, timeout=timeout)
                if response.status_code < 400:
                    return response.json()
                error = f"OpenRouter HTTP {response.status_code}: {response.text[:800]}"
                if response.status_code not in {408, 409, 429, 500, 502, 503, 504}:
                    break
            except (requests.RequestException, ValueError) as exc:
                error = str(exc)
            if attempt < 4:
                delay = min(30, 2 ** attempt) + attempt * 0.37
                log(f"  请求暂时失败，{delay:.1f} 秒后重试")
                time.sleep(delay)
        raise RuntimeError(error)

    def chat(self, *, stage: str, model: str, messages: list[dict[str, Any]],
             max_tokens: int, reasoning: str = "high", temperature: float = 0.15,
             schema: dict[str, Any] | None = None) -> Any:
        key_data = {"pipeline": PIPELINE_VERSION, "stage": stage, "model": model,
                    "messages": messages, "max_tokens": max_tokens, "reasoning": reasoning,
                    "temperature": temperature, "schema": schema}
        key = stable_hash(key_data)
        cache = self.cache_dir / "llm" / stage / f"{key}.json"
        if cache.exists():
            stored = json.loads(cache.read_text(encoding="utf-8"))
            log(f"  使用缓存：{stage}")
            return stored["parsed"]
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning": {"effort": reasoning, "exclude": True},
            "provider": {"require_parameters": True},
        }
        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": re.sub(r"[^a-zA-Z0-9_]", "_", stage)[:60],
                                "strict": True, "schema": schema},
            }
        response = self._post("/chat/completions", payload)
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"模型响应格式异常：{json.dumps(response, ensure_ascii=False)[:800]}") from exc
        parsed = parse_json_text(content) if schema else strip_fence(content)
        atomic_json(cache, {"schema_version": SCHEMA_VERSION, "key": key,
                            "model": model, "parsed": parsed,
                            "usage": response.get("usage", {})})
        return parsed

    def transcribe(self, media: Path, *, model: str, workers: int, chunk_seconds: int,
                   overlap_seconds: int) -> list[dict[str, Any]]:
        duration = float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "default=noprint_wrappers=1:nokey=1", str(media)]))
        step = chunk_seconds - overlap_seconds
        spans: list[tuple[int, float, float]] = []
        start = 0.0
        while start < duration:
            spans.append((len(spans), start, min(duration, start + chunk_seconds)))
            start += step

        def one(item: tuple[int, float, float]) -> dict[str, Any]:
            index, begin, end = item
            fingerprint = stable_hash({"media": file_fingerprint(media), "model": model,
                                       "start": begin, "end": end})
            cache = self.cache_dir / "stt" / f"{index:04d}-{fingerprint[:16]}.json"
            if cache.exists():
                return json.loads(cache.read_text(encoding="utf-8"))
            with tempfile.TemporaryDirectory(prefix="tech-blog-stt-") as temp:
                audio = Path(temp) / "part.mp3"
                run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(begin),
                     "-t", str(end - begin), "-i", str(media), "-vn", "-ac", "1", "-ar", "16000",
                     "-b:a", "32k", str(audio)])
                payload = {"model": model,
                           "input_audio": {"data": base64.b64encode(audio.read_bytes()).decode("ascii"),
                                           "format": "mp3"},
                           "language": "zh", "temperature": 0}
                result = self._post("/audio/transcriptions", payload, timeout=420)
            parsed = {"start": round(begin, 3), "end": round(end, 3),
                      "text": str(result.get("text", "")).strip()}
            atomic_json(cache, parsed)
            return parsed

        result: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(one, span) for span in spans]
            for number, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result.append(future.result())
                log(f"  转写进度：{number}/{len(spans)}")
        return sorted(result, key=lambda x: x["start"])


def ingest_pdf(pdf: Path, root: Path) -> list[dict[str, Any]]:
    pages_json = root / "cache" / "pdf_pages.json"
    assets = root / "assets" / "slides"
    vision = root / "cache" / "vision_slides"
    assets.mkdir(parents=True, exist_ok=True)
    vision.mkdir(parents=True, exist_ok=True)
    fingerprint = file_fingerprint(pdf)
    if pages_json.exists():
        stored = json.loads(pages_json.read_text(encoding="utf-8"))
        if stored.get("fingerprint") == fingerprint and all((assets / f"slide-{p['page']:02d}.png").exists()
                                                             for p in stored["pages"]):
            return stored["pages"]
    document = fitz.open(pdf)
    pages: list[dict[str, Any]] = []
    for number, page in enumerate(document, 1):
        text = page.get_text("text").strip()
        pix = page.get_pixmap(matrix=fitz.Matrix(2.2, 2.2), alpha=False)
        final_image = assets / f"slide-{number:02d}.png"
        pix.save(final_image)
        vision_image = vision / f"slide-{number:02d}.jpg"
        with Image.open(final_image) as source:
            source.thumbnail((1100, 700), Image.Resampling.LANCZOS)
            source.convert("RGB").save(vision_image, "JPEG", quality=78, optimize=True)
        pages.append({"page": number, "text": text,
                      "image": f"assets/slides/slide-{number:02d}.png",
                      "vision_image": str(vision_image)})
        log(f"  PDF 页面：{number}/{len(document)}")
    document.close()
    atomic_json(pages_json, {"schema_version": SCHEMA_VERSION,
                             "fingerprint": fingerprint, "pages": pages})
    return pages


SLIDE_SCHEMA = {
    "type": "object",
    "properties": {
        "pages": {"type": "array", "items": {"type": "object", "properties": {
            "page": {"type": "integer"}, "title": {"type": "string"},
            "purpose": {"type": "string"}, "diagram_description": {"type": "string"},
            "key_points": {"type": "array", "items": {"type": "string"}},
            "numbers": {"type": "array", "items": {"type": "string"}},
            "caveats": {"type": "array", "items": {"type": "string"}},
            "visual_value": {"type": "string", "enum": ["high", "medium", "low"]},
        }, "required": ["page", "title", "purpose", "diagram_description", "key_points",
                        "numbers", "caveats", "visual_value"], "additionalProperties": False}},
    }, "required": ["pages"], "additionalProperties": False,
}


def analyze_slide_batch(client: OpenRouter, pages: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": """你是技术演讲材料分析员。逐页阅读图片和提取文本，输出严格 JSON。重点识别：页面在故事线中的作用、图中数据流/内存布局/时序、准确数字、实验限定条件、容易误读之处、是否值得在博客中使用。不得根据常识补全看不清的数字。"""}]
    for page in pages:
        raw = Path(page["vision_image"]).read_bytes()
        data = base64.b64encode(raw).decode("ascii")
        content.append({"type": "text", "text": f"第 {page['page']} 页，文本层：\n{page['text']}"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}})
    result = client.chat(stage=f"slide-vision-{pages[0]['page']:02d}-{pages[-1]['page']:02d}",
                         model=model, messages=[{"role": "user", "content": content}],
                         max_tokens=8000, reasoning="medium", schema=SLIDE_SCHEMA)
    return result["pages"]


EVIDENCE_SCHEMA = {
    "type": "object", "properties": {
        "document_title": {"type": "string"}, "one_sentence_thesis": {"type": "string"},
        "glossary": {"type": "array", "items": {"type": "object", "properties": {
            "term": {"type": "string"}, "definition": {"type": "string"},
            "heard_as": {"type": "array", "items": {"type": "string"}},
        }, "required": ["term", "definition", "heard_as"], "additionalProperties": False}},
        "facts": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "claim": {"type": "string"},
            "source_refs": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "notes": {"type": "string"},
        }, "required": ["id", "claim", "source_refs", "confidence", "notes"],
            "additionalProperties": False}},
        "storyline": {"type": "array", "items": {"type": "string"}},
        "warnings": {"type": "array", "items": {"type": "string"}},
    }, "required": ["document_title", "one_sentence_thesis", "glossary", "facts",
                     "storyline", "warnings"], "additionalProperties": False,
}


OUTLINE_SCHEMA = {
    "type": "object", "properties": {
        "title": {"type": "string"}, "subtitle": {"type": "string"},
        "audience": {"type": "string"},
        "prerequisites": {"type": "array", "items": {"type": "string"}},
        "reading_goals": {"type": "array", "items": {"type": "string"}},
        "sections": {"type": "array", "minItems": 6, "maxItems": 9, "items": {
            "type": "object", "properties": {
                "id": {"type": "string"}, "title": {"type": "string"},
                "question": {"type": "string"}, "answer": {"type": "string"},
                "prerequisites": {"type": "array", "items": {"type": "string"}},
                "transcript_indices": {"type": "array", "items": {"type": "integer"}},
                "slide_pages": {"type": "array", "items": {"type": "integer"}},
                "figure_pages": {"type": "array", "items": {"type": "integer"}},
                "key_fact_ids": {"type": "array", "items": {"type": "string"}},
                "writing_brief": {"type": "string"}, "transition_reason": {"type": "string"},
                "target_chars": {"type": "integer"},
            }, "required": ["id", "title", "question", "answer", "prerequisites",
                            "transcript_indices", "slide_pages", "figure_pages", "key_fact_ids",
                            "writing_brief", "transition_reason", "target_chars"],
               "additionalProperties": False}},
        "conclusion_points": {"type": "array", "items": {"type": "string"}},
    }, "required": ["title", "subtitle", "audience", "prerequisites", "reading_goals",
                     "sections", "conclusion_points"], "additionalProperties": False,
}


REVIEW_SCHEMA = {
    "type": "object", "properties": {
        "pass": {"type": "boolean"}, "summary": {"type": "string"},
        "issues": {"type": "array", "items": {"type": "object", "properties": {
            "severity": {"type": "string", "enum": ["P0", "P1", "P2", "P3"]},
            "category": {"type": "string"}, "location": {"type": "string"},
            "problem": {"type": "string"}, "evidence": {"type": "string"},
            "suggested_fix": {"type": "string"},
        }, "required": ["severity", "category", "location", "problem", "evidence", "suggested_fix"],
            "additionalProperties": False}},
    }, "required": ["pass", "summary", "issues"], "additionalProperties": False,
}


def transcript_lines(transcript: list[dict[str, Any]], indices: list[int] | None = None) -> str:
    selected = indices if indices is not None else list(range(len(transcript)))
    rows = []
    for index in selected:
        if 0 <= index < len(transcript):
            item = transcript[index]
            rows.append(f"T{index:02d} [{timestamp(float(item['start']))}-{timestamp(float(item['end']))}] {item['text']}")
    return "\n".join(rows)


def page_lines(pages: list[dict[str, Any]], analyses: list[dict[str, Any]], selected: list[int] | None = None) -> str:
    by_page = {int(item["page"]): item for item in analyses}
    allowed = set(selected) if selected is not None else None
    rows = []
    for page in pages:
        number = int(page["page"])
        if allowed is not None and number not in allowed:
            continue
        rows.append(f"P{number:02d} 文本：{page['text']}\n视觉分析：{json.dumps(by_page.get(number, {}), ensure_ascii=False)}")
    return "\n\n".join(rows)


def load_transcript(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("transcript") or value.get("segments")
    if not isinstance(value, list):
        raise RuntimeError("转写 JSON 必须是数组，或包含 transcript/segments 数组")
    result = []
    for item in value:
        result.append({"start": float(item["start"]), "end": float(item["end"]),
                       "text": str(item["text"]).strip()})
    return result


def find_transcript(media: Path, explicit: Path | None) -> Path | None:
    if explicit:
        return explicit
    candidates = [Path.cwd() / f"{media.stem}_chapters_transcript.json",
                  media.with_name(media.stem + "_chapters_transcript.json")]
    return next((path for path in candidates if path.is_file()), None)


def make_evidence(client: OpenRouter, transcript: list[dict[str, Any]], pages: list[dict[str, Any]],
                  slide_analysis: list[dict[str, Any]], model: str) -> dict[str, Any]:
    prompt = f"""你是技术事实编辑。把视频转写和 PPT 建成证据台账，供后续作者使用。

硬规则：
1. 视频转写存在同音词错误；以 PPT 的术语拼写为准，但不得改写演讲含义。
2. 每个事实给唯一 Fxx 编号和来源：Txx 表示转写切片，Pxx 表示 PPT 页。
3. 数字、倍数、性能和因果结论必须保留适用条件；只有一个来源时如实写明。
4. 参考文不是事实来源。不要使用外部常识填空。
5. 提炼一条因果故事线，不按页码机械罗列。

PPT：
{page_lines(pages, slide_analysis)}

转写：
{transcript_lines(transcript)}
"""
    return client.chat(stage="evidence-ledger", model=model,
                       messages=[{"role": "user", "content": prompt}], max_tokens=18000,
                       reasoning="high", schema=EVIDENCE_SCHEMA)


def make_outline(client: OpenRouter, evidence: dict[str, Any], transcript: list[dict[str, Any]],
                 pages: list[dict[str, Any]], slide_analysis: list[dict[str, Any]], model: str,
                 max_sections: int) -> dict[str, Any]:
    prompt = f"""你是资深技术主编。为一篇图文技术博客规划 6-{max_sections} 个核心章节。

{STYLE_PROFILE}
{REFERENCE_SAFETY}

规划规则：
- 主线必须是一条因果链：上一节暴露的问题，由下一节解决或深化。
- 不要照视频时间、PPT 页码逐页翻译；把问答中的支线放入最相关章节或局限。
- 每节列出唯一可用的 Txx、Pxx、Fxx；writer 只能看到这些证据。
- 全文选择 8-12 张高信息量 PPT 图；同一页最多使用一次。低信息页用表格/文字重构。
- 性能数字必须带硬件、batch、并行度或页面明确写出的限定条件。
- 目标总长度 12000-18000 中文字符，单节 target_chars 合理分配。

证据台账：
{json.dumps(evidence, ensure_ascii=False)}

可用转写切片：0-{len(transcript)-1}
PPT 页：1-{len(pages)}
页面视觉摘要：
{json.dumps(slide_analysis, ensure_ascii=False)}
"""
    outline = client.chat(stage="outline", model=model,
                          messages=[{"role": "user", "content": prompt}], max_tokens=12000,
                          reasoning="high", schema=OUTLINE_SCHEMA)
    if len(outline["sections"]) > max_sections:
        outline["sections"] = outline["sections"][:max_sections]
    return outline


def write_section(client: OpenRouter, section: dict[str, Any], outline: dict[str, Any],
                  evidence: dict[str, Any], transcript: list[dict[str, Any]],
                  pages: list[dict[str, Any]], slide_analysis: list[dict[str, Any]], model: str) -> str:
    facts = [fact for fact in evidence["facts"] if fact["id"] in set(section["key_fact_ids"])]
    prompt = f"""你是一位中文系统工程技术作者。只写大纲中的一个章节，直接输出 Markdown，从 `##` 标题开始。

{STYLE_PROFILE}
{REFERENCE_SAFETY}

全局文章：{outline['title']}
本节计划：{json.dumps(section, ensure_ascii=False)}
允许事实：{json.dumps(facts, ensure_ascii=False)}
统一术语：{json.dumps(evidence['glossary'], ensure_ascii=False)}

允许的 PPT 证据：
{page_lines(pages, slide_analysis, section['slide_pages'])}

允许的视频证据：
{transcript_lines(transcript, section['transcript_indices'])}

写作硬约束：
1. 仅依据上面的证据；不要编造源码函数、版本、数字或实验结论。
2. 对性能数字写清页面提供的限定条件；证据不足时明确写“演讲材料未给出……”。
3. 目标约 {section['target_chars']} 中文字符。围绕“{section['question']}”建立问题—机制—例子—边界闭环。
4. 对 figure_pages 中每一页，在最相关位置使用一次：`![准确的中文alt](assets/slides/slide-XX.png)`，下一行写斜体图注并注明“来源：演讲 PPT，第 X 页”。正文必须解释图中具体元素。
5. 不输出证据编号 T/P/F，不写“根据材料”式流水账；来源信息放在自然语言限定和图注中。
6. 可用 Markdown 表格、列表和简短公式；不生成 Mermaid，不引用不存在的图片。
7. 不写全文引言或全文总结，不重复其他章节。
"""
    return client.chat(stage=f"section-{section['id']}", model=model,
                       messages=[{"role": "user", "content": prompt}],
                       max_tokens=10000, reasoning="high", temperature=0.2)


def global_edit(client: OpenRouter, draft: str, outline: dict[str, Any], evidence: dict[str, Any],
                model: str, stage: str = "global-edit", issues: list[dict[str, Any]] | None = None) -> str:
    issue_text = "" if not issues else "\n必须定点修复的审稿问题：\n" + json.dumps(issues, ensure_ascii=False)
    prompt = f"""你是技术出版物总编。把分节草稿编辑成一篇可以直接发布的中文 Markdown 长文。

{STYLE_PROFILE}
{REFERENCE_SAFETY}

大纲：{json.dumps(outline, ensure_ascii=False)}
事实台账：{json.dumps(evidence, ensure_ascii=False)}
{issue_text}

草稿：
{draft}

编辑硬规则：
- 直接输出完整 Markdown，不要代码围栏，不要解释编辑过程。
- 保留且只使用草稿里真实存在的 `assets/slides/slide-XX.png` 图片路径，不新增虚构资产。
- 写一个有信息量的 H1 标题、短导语、适读人群/前置知识、阅读目标；主体保持 6-9 节因果顺序；结尾用 5-8 条结论和明确局限收束。
- 删除分节之间的重复，补足过渡；术语首次出现解释，全文拼写统一。
- 不引入事实台账外的新数字、源码细节、实验结果或外部资料。
- 对 460 tok/s、倍数、600 倍、6k+ block、19.5% 等数字保留精确适用条件，不能泛化。
- 每张图在图前说明“为什么看这张图”，图后解释关键箭头/模块/数据；图注保留 PPT 页码。
- 文风独立，不模仿参考作者的标志性表达。
"""
    return client.chat(stage=stage, model=model, messages=[{"role": "user", "content": prompt}],
                       max_tokens=30000, reasoning="high", temperature=0.15)


def review_article(client: OpenRouter, article: str, outline: dict[str, Any], evidence: dict[str, Any],
                   model: str) -> dict[str, Any]:
    prompt = f"""你是独立技术审稿人。逐项检查文章，但只输出严格 JSON。

检查维度：事实和数字是否有台账支持；限定条件是否丢失；术语是否统一；因果是否跳跃；图片是否被正文解读；是否逐页翻译；是否有空泛段落；是否疑似模仿参考作者；Markdown 是否可发布。
P0=严重虚构/错误，P1=关键事实或逻辑问题，P2=明显影响质量，P3=可选润色。
只有没有 P0/P1 且 P2 很少时 pass 才能为 true。问题必须指明章节和可执行修复。

大纲：{json.dumps(outline, ensure_ascii=False)}
事实台账：{json.dumps(evidence, ensure_ascii=False)}
文章：
{article}
"""
    return client.chat(stage="independent-review", model=model,
                       messages=[{"role": "user", "content": prompt}], max_tokens=10000,
                       reasoning="high", schema=REVIEW_SCHEMA)


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def chinese_ngrams(text: str, size: int = 8) -> set[str]:
    cleaned = re.sub(r"[^\u4e00-\u9fff]", "", text)
    return {cleaned[i:i + size] for i in range(max(0, len(cleaned) - size + 1))}


def deterministic_qa(article: str, root: Path, reference_html: Path | None,
                     review: dict[str, Any]) -> dict[str, Any]:
    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", article)
    missing = [link for link in links if not (root / link).is_file()]
    placeholders = sorted(set(re.findall(r"(?:TODO|TBD|待补充|\{\{[^}]+\}\})", article, re.I)))
    overlap: list[str] = []
    if reference_html and reference_html.is_file():
        parser = TextExtractor()
        parser.feed(reference_html.read_text(encoding="utf-8", errors="ignore"))
        overlap = sorted(chinese_ngrams(article) & chinese_ngrams(html.unescape("".join(parser.parts))))
        overlap = [item for item in overlap if not re.search(r"异步推理|技术博客|系统架构", item)][:100]
    headings = re.findall(r"^(#{1,6})\s+(.+)$", article, re.M)
    heading_jumps = []
    for left, right in zip(headings, headings[1:]):
        if len(right[0]) > len(left[0]) + 1:
            heading_jumps.append(f"{left[1]} -> {right[1]}")
    severe = [issue for issue in review.get("issues", []) if issue.get("severity") in {"P0", "P1"}]
    passed = not missing and not placeholders and not heading_jumps and not severe and len(article) >= 7000
    return {
        "schema_version": SCHEMA_VERSION, "pass": passed, "article_chars": len(article),
        "image_count": len(links), "missing_images": missing, "placeholders": placeholders,
        "heading_jumps": heading_jumps, "reference_8gram_overlaps": overlap,
        "review_pass": review.get("pass", False), "severe_review_issues": severe,
        "notes": ["8-gram 命中用于人工复核；技术术语造成的命中不自动判定抄袭。"],
    }


def package_markdown_assets(article: str, source_root: Path, final_dir: Path) -> list[str]:
    """Copy referenced local images beside the final Markdown for portable preview."""
    copied: list[str] = []
    links = sorted(set(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", article)))
    for link in links:
        relative = Path(link)
        if relative.is_absolute() or ".." in relative.parts:
            continue
        source = source_root / relative
        if not source.is_file():
            continue
        destination = final_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(link)
    return copied


def main() -> int:
    global STYLE_PROFILE
    parser = argparse.ArgumentParser(description="把技术视频和 PPT/PDF 生成图文并茂的 Markdown 技术博客")
    parser.add_argument("media", type=Path, help="技术视频或音频")
    parser.add_argument("slides", type=Path, help="对应 PPT 导出的 PDF")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("tech_blog_output"))
    parser.add_argument("--transcript-json", type=Path, help="复用现有带时间戳转写 JSON")
    parser.add_argument("--style-profile", type=Path,
                        default=Path("references/mengyuan_async_llm/STYLE_PROFILE.md"))
    parser.add_argument("--reference-html", type=Path,
                        default=Path("references/mengyuan_async_llm/original.html"))
    parser.add_argument("--analyst-model", default=DEFAULT_ANALYST)
    parser.add_argument("--writer-model", default=DEFAULT_WRITER)
    parser.add_argument("--critic-model", default=DEFAULT_CRITIC)
    parser.add_argument("--stt-model", default=DEFAULT_STT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-sections", type=int, default=8)
    parser.add_argument("--skip-slide-vision", action="store_true")
    parser.add_argument("--no-repair", action="store_true")
    args = parser.parse_args()

    if not args.media.is_file():
        parser.error(f"找不到媒体文件：{args.media}")
    if not args.slides.is_file():
        parser.error(f"找不到 PDF：{args.slides}")
    if args.slides.suffix.lower() != ".pdf":
        parser.error("当前版本请先把 PPT 导出为 PDF")
    if not 6 <= args.max_sections <= 9:
        parser.error("--max-sections 必须在 6 到 9 之间")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        parser.error("请设置环境变量 OPENROUTER_API_KEY；脚本不会保存明文密钥")
    if args.style_profile.is_file():
        STYLE_PROFILE = args.style_profile.read_text(encoding="utf-8")

    root = args.output_dir.resolve()
    for folder in ["cache", "drafts", "final", "assets/slides"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION,
        "media": file_fingerprint(args.media), "slides": file_fingerprint(args.slides),
        "models": {"analyst": args.analyst_model, "writer": args.writer_model,
                   "critic": args.critic_model, "stt": args.stt_model},
        "reference": str(args.reference_html.resolve()) if args.reference_html.exists() else None,
    }
    atomic_json(root / "run.json", manifest)
    client = OpenRouter(api_key, root / "cache")

    log("[1/7] 提取并渲染 PPT")
    pages = ingest_pdf(args.slides, root)

    transcript_path = find_transcript(args.media, args.transcript_json)
    if transcript_path:
        log(f"[2/7] 复用现有转写：{transcript_path}")
        transcript = load_transcript(transcript_path)
    else:
        log("[2/7] 没有发现现成转写，开始带重叠并行转写")
        transcript = client.transcribe(args.media, model=args.stt_model, workers=args.workers,
                                       chunk_seconds=180, overlap_seconds=15)
    atomic_json(root / "cache" / "transcript.json", transcript)

    log("[3/7] 多模态分析 PPT 页面")
    if args.skip_slide_vision:
        slide_analysis = [{"page": p["page"], "title": "", "purpose": p["text"][:200],
                           "diagram_description": "未启用视觉分析", "key_points": [],
                           "numbers": [], "caveats": [], "visual_value": "medium"} for p in pages]
    else:
        batches = [pages[i:i + 4] for i in range(0, len(pages), 4)]
        slide_analysis = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(analyze_slide_batch, client, batch, args.analyst_model) for batch in batches]
            for number, future in enumerate(concurrent.futures.as_completed(futures), 1):
                slide_analysis.extend(future.result())
                log(f"  视觉分析：{number}/{len(batches)}")
        slide_analysis.sort(key=lambda x: x["page"])
    atomic_json(root / "cache" / "slide_analysis.json", slide_analysis)

    log("[4/7] 建立事实台账和因果大纲")
    evidence = make_evidence(client, transcript, pages, slide_analysis, args.analyst_model)
    atomic_json(root / "cache" / "evidence.json", evidence)
    outline = make_outline(client, evidence, transcript, pages, slide_analysis,
                           args.analyst_model, args.max_sections)
    atomic_json(root / "cache" / "outline.json", outline)

    log("[5/7] 并行撰写各章节")
    sections: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(write_section, client, section, outline, evidence, transcript,
                               pages, slide_analysis, args.writer_model): section
                   for section in outline["sections"]}
        for number, future in enumerate(concurrent.futures.as_completed(futures), 1):
            section = futures[future]
            text = future.result()
            sections[section["id"]] = text
            atomic_text(root / "drafts" / f"section-{section['id']}.md", text)
            log(f"  章节草稿：{number}/{len(futures)}")
    ordered = "\n\n".join(sections[section["id"]] for section in outline["sections"])
    atomic_text(root / "drafts" / "assembled.md", ordered)

    log("[6/7] 全局编辑并独立审稿")
    edited = global_edit(client, ordered, outline, evidence, args.writer_model)
    atomic_text(root / "drafts" / "edited.md", edited)
    review = review_article(client, edited, outline, evidence, args.critic_model)
    atomic_json(root / "cache" / "review.json", review)
    actionable = [issue for issue in review["issues"] if issue["severity"] in {"P0", "P1", "P2"}]
    final = edited
    if actionable and not args.no_repair:
        log(f"  发现 {len(actionable)} 个需要修复的问题，执行定点修复")
        final = global_edit(client, edited, outline, evidence, args.writer_model,
                            stage="review-repair", issues=actionable)

    log("[7/7] 确定性检查并写出成稿")
    final_path = root / "final" / "blog.md"
    atomic_text(final_path, final.rstrip() + "\n")
    packaged_assets = package_markdown_assets(final, root, final_path.parent)
    qa = deterministic_qa(final, final_path.parent,
                          args.reference_html if args.reference_html.exists() else None, review)
    qa["packaged_assets"] = packaged_assets
    atomic_json(root / "final" / "qa-report.json", qa)
    atomic_json(root / "final" / "sources.json", {
        "media": str(args.media.resolve()), "slides": str(args.slides.resolve()),
        "transcript": str(transcript_path.resolve()) if transcript_path else "generated",
        "reference_style_only": str(args.reference_html.resolve()) if args.reference_html.exists() else None,
        "models": manifest["models"],
    })
    docx_path = root / "final" / "blog.docx"
    markdown_to_docx(
        final_path,
        docx_path,
        source_label=f"配套材料：{args.slides.name}",
    )
    qa["docx"] = audit_docx(docx_path)
    if not qa["docx"]["pass"]:
        qa["pass"] = False
    atomic_json(root / "final" / "qa-report.json", qa)
    status = "通过" if qa["pass"] else "需查看 qa-report.json"
    log(f"完成（{status}）：{final_path}\nDOCX：{docx_path}")
    return 0 if not qa["missing_images"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("已取消；再次运行会从阶段缓存继续。")
        raise SystemExit(130)
    except Exception as exc:
        log(f"错误：{exc}")
        raise SystemExit(1)
