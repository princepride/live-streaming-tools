#!/usr/bin/env python3
"""Create balanced Bilibili-style chapters for a video or audio file."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

import stt


API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_STT_MODEL = "openai/whisper-large-v3"
DEFAULT_CHAPTER_MODEL = "google/gemini-3.1-pro-preview"


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Command failed")
    return result.stdout.strip()


def media_duration(path: Path) -> float:
    output = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    return float(output)


def stamp(seconds: float) -> str:
    value = max(0, round(seconds))
    hours, rest = divmod(value, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def shorten_title(title: str, limit: int) -> str:
    """Shorten a title without cutting an ASCII technical term in half."""
    value = re.sub(r"\s+", " ", title.strip())
    replacements = {
        "Hybrid Memory Allocator": "混合内存管理",
        "System Prompt": "系统提示词",
        "Partial Cache Hit": "部分缓存命中",
        "Partial Cache": "部分缓存",
        "Prefix Cache": "前缀缓存",
        "Tensor Parallel": "张量并行",
        "Reduce Scatter": "规约分散",
        "Kernel": "内核",
    }
    for source, target in replacements.items():
        value = re.sub(re.escape(source), target, value, flags=re.I)
    value = re.sub(r"^(.{1,8})对(.{1,8})的挑战与(.+)$", r"\1\3挑战", value)
    value = value.replace("详细解析", "解析").replace("详细讲解", "讲解")
    value = value.replace("实现细节", "细节").replace("性能优化", "优化")
    if len(value) <= limit:
        return value.rstrip("：:、，,；;与和及的")

    # Treat each ASCII word as an indivisible token; CJK characters stay compact.
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9+._/-]*|.", value)
    kept: list[str] = []
    used = 0
    for token in tokens:
        if used + len(token) > limit:
            break
        kept.append(token)
        used += len(token)
    result = "".join(kept).rstrip(" ：:、，,；;与和及的")
    return result or value[:limit]


def request_json(url: str, headers: dict, payload: dict, timeout: int, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if response.status_code < 400:
                return response.json()
            detail = response.text[:500]
            if response.status_code not in {408, 409, 429, 500, 502, 503, 504}:
                raise RuntimeError(f"OpenRouter HTTP {response.status_code}: {detail}")
            error = f"OpenRouter HTTP {response.status_code}: {detail}"
        except (requests.RequestException, ValueError) as exc:
            error = str(exc)
        if attempt + 1 < retries:
            delay = 2 ** attempt
            log(f"  请求失败，{delay} 秒后重试：{error}")
            time.sleep(delay)
    raise RuntimeError(error)


def parse_json_content(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I | re.S)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.S)
        if not match:
            raise RuntimeError("章节模型没有返回可解析的 JSON")
        return json.loads(match.group(0))


def make_chapters(transcript: list[dict], duration: float, headers: dict, model: str,
                  min_minutes: float, max_minutes: float, max_chapters: int,
                  max_title_chars: int) -> list[dict]:
    transcript_text = "\n".join(
        f"切片 {i} [{stamp(x['start'])} - {stamp(x['end'])}] {x['text']}"
        for i, x in enumerate(transcript)
    )
    slice_count = len(transcript)
    # Fine-mode slices are short, uneven sentences; use the average length so the
    # min/max-slice guidance stays stable regardless of any single segment.
    typical_slice_seconds = max(1, round(duration / max(1, slice_count)))
    min_slices = max(1, math.ceil(min_minutes * 60 / typical_slice_seconds))
    # The count cap may require slightly longer chapters than max_minutes.
    max_slices = max(
        min_slices,
        math.floor(max_minutes * 60 / typical_slice_seconds),
        math.ceil(slice_count / max_chapters),
    )
    prompt = f"""你是一位资深中文视频编辑。根据下面按时间切片的转写，制作适合 B 站的章节。

要求：
1. 完整覆盖全部 {slice_count} 个切片，连续、无重叠、无空档。
1.1 最多生成 {max_chapters} 个章节，这是硬性限制；本视频建议生成 8-{max_chapters} 章。
2. 章节颗粒度适中：每章必须包含 {min_slices}-{max_slices} 个切片（约 {min_minutes:g}-{max_minutes:g} 分钟）。最后剩余不足 {min_slices} 个切片时必须并入上一章，禁止单独成章。
3. 用 start_index 和 end_index 表示切片范围，start_index 包含、end_index 不包含。第一章必须从 0 开始，最后一章必须到 {slice_count} 结束。
4. title 是准确、具体、简短的中文标题，必须不超过 {max_title_chars} 个字符（标点、英文和数字也各算一个字符），不要写“第一部分”等空泛标题。
5. summary 用一句中文说明这段讲了什么（建议 25-60 字），不要编造转写中没有的信息。
6. 相邻切片仍在讲同一件事时必须合并；不要按固定间隔机械分章。
7. 在满足每章 {min_slices}-{max_slices} 个切片的硬约束下，边界以话题、讲解阶段或问答主题的真实变化为依据。章节长度应自然有变化。

只返回 JSON，格式为：
{{"chapters":[{{"start_index":0,"end_index":2,"title":"标题","summary":"内容概述"}}]}}

转写：
{transcript_text}
"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.15,
        "max_tokens": 5000,
        "reasoning": {"effort": "medium", "exclude": True},
        "provider": {"require_parameters": True},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "bilibili_chapters",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "chapters": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": max_chapters,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "start_index": {"type": "integer", "minimum": 0, "maximum": slice_count - 1},
                                    "end_index": {"type": "integer", "minimum": 1, "maximum": slice_count},
                                    "title": {"type": "string", "minLength": 1, "maxLength": max_title_chars},
                                    "summary": {"type": "string", "minLength": 1},
                                },
                                "required": ["start_index", "end_index", "title", "summary"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["chapters"],
                    "additionalProperties": False,
                },
            },
        },
    }
    result = request_json(f"{API_BASE}/chat/completions", headers, payload, timeout=300)
    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"章节接口返回异常：{json.dumps(result, ensure_ascii=False)[:500]}") from exc
    parsed = parse_json_content(content)
    chapters = parsed.get("chapters", []) if isinstance(parsed, dict) else parsed
    if not isinstance(chapters, list):
        raise RuntimeError("章节模型返回的 chapters 不是数组")
    if not chapters:
        raise RuntimeError("模型没有生成章节")

    cleaned = []
    previous_end = 0
    for position, chapter in enumerate(chapters):
        # The model decides thematic end points. Programmatic normalization makes
        # the ranges exhaustive even when it repeats/omits a neighboring index.
        end_index = int(chapter["end_index"])
        if position == len(chapters) - 1:
            end_index = slice_count
        end_index = min(slice_count, end_index)
        start_index = previous_end
        if end_index <= start_index:
            continue
        start = round(transcript[start_index]["start"])
        end = round(duration) if end_index == slice_count else round(transcript[end_index]["start"])
        cleaned.append({
            "start": start, "end": end,
            "title": shorten_title(str(chapter["title"]), max_title_chars),
            "summary": str(chapter["summary"]).strip(),
        })
        previous_end = end_index
    if not cleaned:
        raise RuntimeError("模型没有生成有效章节")
    if len(cleaned) > 1 and cleaned[-1]["end"] - cleaned[-1]["start"] < min_minutes * 30:
        tail = cleaned.pop()
        cleaned[-1]["end"] = tail["end"]
        cleaned[-1]["summary"] = cleaned[-1]["summary"].rstrip("。") + "；最后" + tail["summary"]
    # Enforce the user's absolute chapter-count limit even if a model ignores it.
    while len(cleaned) > max_chapters:
        pair = min(range(len(cleaned) - 1), key=lambda i: cleaned[i + 1]["end"] - cleaned[i]["start"])
        left, right = cleaned[pair], cleaned[pair + 1]
        left["end"] = right["end"]
        left["title"] = shorten_title(
            left["title"].rstrip("解析详解概览") + "与" + right["title"],
            max_title_chars,
        )
        left["summary"] = left["summary"].rstrip("。") + "；同时，" + right["summary"]
        cleaned.pop(pair + 1)
    if cleaned[0]["start"] != 0 or cleaned[-1]["end"] != round(duration):
        raise RuntimeError("章节没有覆盖完整视频，请重试或更换章节模型")
    for left, right in zip(cleaned, cleaned[1:]):
        if left["end"] != right["start"]:
            raise RuntimeError("章节之间存在空档或重叠，请重试")
    return cleaned


def write_outputs(output: Path, media: Path, duration: float, transcript: list[dict], chapters: list[dict]) -> None:
    data = {"media": str(media.resolve()), "duration": duration, "chapters": chapters}
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    text_path = output.with_suffix(".txt")
    readable = [
        f"{stamp(c['start'])} - {stamp(c['end'])}  {c['title']}\n{c['summary']}"
        for c in chapters
    ]
    bilibili = "\n".join(f"{stamp(c['start'])} {c['title']}" for c in chapters)
    text_path.write_text("\n\n".join(readable) + "\n\nB站时间点格式：\n" + bilibili + "\n", encoding="utf-8")
    transcript_path = output.with_name(output.stem + "_transcript.json")
    transcript_path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"完成：\n  {output}\n  {text_path}\n  {transcript_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="用 OpenRouter 为视频或音频自动生成 B 站章节")
    parser.add_argument("media", type=Path, help="视频或音频文件路径")
    parser.add_argument("-o", "--output", type=Path, help="输出 JSON 路径")
    parser.add_argument("--chunk-seconds", type=int, default=240, help="转写切片秒数（默认 240）")
    parser.add_argument("--min-minutes", type=float, default=6, help="期望最短章节分钟数（默认 6）")
    parser.add_argument("--max-minutes", type=float, default=10, help="期望最长章节分钟数（默认 10）")
    parser.add_argument("--max-chapters", type=int, default=10, help="章节数量上限（默认 10）")
    parser.add_argument("--max-title-chars", type=int, default=16, help="章节标题字符上限（默认 16）")
    parser.add_argument("--workers", type=int, default=3, help="并行转写数（默认 3）")
    parser.add_argument("--language", default="zh", help="音频语言 ISO 代码（默认 zh）")
    parser.add_argument("--stt-backend", default="openrouter",
                        choices=["openrouter", "groq", "openai", "local"],
                        help="转写后端（默认 openrouter，共用 stt.py）")
    parser.add_argument("--stt-model", default=DEFAULT_STT_MODEL)
    parser.add_argument("--chapter-model", default=DEFAULT_CHAPTER_MODEL)
    args = parser.parse_args()

    if not args.media.is_file():
        parser.error(f"找不到媒体文件：{args.media}")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        parser.error("需要先安装 ffmpeg，并确保 ffmpeg/ffprobe 在 PATH 中")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        parser.error("请先设置环境变量 OPENROUTER_API_KEY（脚本不会读取或保存明文密钥）")
    if args.chunk_seconds < 60:
        parser.error("--chunk-seconds 不能小于 60")
    if args.min_minutes <= 0 or args.max_minutes < args.min_minutes:
        parser.error("章节分钟数必须满足 0 < --min-minutes <= --max-minutes")
    if args.max_chapters < 1:
        parser.error("--max-chapters 必须至少为 1")
    if args.max_title_chars < 4:
        parser.error("--max-title-chars 不能小于 4")

    output = args.output or Path.cwd() / f"{args.media.stem}_chapters.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.stt_model)
    cache_dir = output.parent / f".{output.stem}_cache_{args.chunk_seconds}_{safe_model}_{args.language}_fine"
    cache_dir.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "X-Title": "Bilibili Auto Chapters"}
    duration = media_duration(args.media)
    log(f"媒体时长 {stamp(duration)}，开始转写（后端 {args.stt_backend}，模型 {args.stt_model}）")
    transcript = stt.transcribe(
        args.media, backend=args.stt_backend, model=args.stt_model,
        language=args.language, chunk_seconds=args.chunk_seconds, mode="fine",
        cache_dir=cache_dir, workers=args.workers,
    )
    if not transcript:
        parser.error("转写结果为空，请检查音频与 API 配置")
    log("正在按主题合并并生成章节……")
    chapters = make_chapters(
        transcript, duration, headers, args.chapter_model,
        args.min_minutes, args.max_minutes, args.max_chapters, args.max_title_chars,
    )
    write_outputs(output, args.media, duration, transcript, chapters)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("已取消；再次运行会从缓存继续。")
        raise SystemExit(130)
    except Exception as exc:
        log(f"错误：{exc}")
        raise SystemExit(1)
