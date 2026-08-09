"""Stage 2: find golden-quote ("金句") highlight clips in a transcript.

Reads <stem>_highlights_transcript.json (Stage 1) and asks an LLM to propose
several self-contained ~60s clips, each with a hook quote, precise start/end
timestamps (snapped to transcript segments), a score, a reason, and the full
dialogue. Long transcripts are processed in overlapping windows and merged.

Backends:
  --backend openrouter  (default)  OPENROUTER_API_KEY, model gemini-3.1-pro-preview
  --backend local                  any OpenAI-compatible server (vLLM/Ollama)

Outputs: <stem>_highlights.json  and  <stem>_highlights.md
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    chat_json, load_env_file, log, read_json, stamp, write_json,
)

BACKENDS = {
    "openrouter": ("https://openrouter.ai/api/v1", "google/gemini-3.1-pro-preview",
                   "OPENROUTER_API_KEY"),
    "local": ("http://localhost:8000/v1", "local-model", "LOCAL_API_KEY"),
}

PROMPT_TEMPLATE = """你是一个短视频剪辑助手。下面是一段技术直播/访谈的转录，按句给出，每行格式：
[序号] mm:ss-mm:ss 文本

请从中挑选出最适合做独立短片段（短视频/切片）的「金句」时刻。要求：
- 每个片段时长在 {min_s}-{max_s} 秒之间（用序号范围保证片段完整，不要从半句开始或结束）。
- 每个片段必须是一个自洽、能独立观看的完整观点或精彩瞬间，开头有钩子，结尾有落点。
- 最多挑选 {max_clips} 个，按精彩/可传播程度从高到低排序。
- hook 是这个片段里最有冲击力、最值得当标题的一句原话（尽量用原文）。
- score 是 0-100 的整数，表示金句/可传播程度。
- reason 用一句中文说明为什么值得剪。

只返回 JSON，格式：
{{"clips":[{{"start_index":12,"end_index":18,"hook":"原话金句","score":88,"reason":"一句话理由"}}]}}
其中 start_index 和 end_index 都是上面的行序号，均为闭区间（end_index 那一句包含在内）。

转录：
{lines}
"""


def build_lines(segments, i0, i1):
    out = []
    for i in range(i0, i1):
        s = segments[i]
        out.append(f"[{i}] {stamp(s['start'])}-{stamp(s['end'])} {s['text']}")
    return "\n".join(out)


def windows(segments, window_s, overlap_s):
    """Yield (start_idx, end_idx_exclusive) covering all segments by time window."""
    if not segments:
        return
    total_end = segments[-1]["end"]
    if total_end <= window_s:
        yield 0, len(segments)
        return
    start_t = 0.0
    n = len(segments)
    i = 0
    while i < n:
        # advance i to first segment whose start >= start_t
        while i < n and segments[i]["start"] < start_t:
            i += 1
        if i >= n:
            break
        j = i
        while j < n and segments[j]["start"] < start_t + window_s:
            j += 1
        yield i, j
        if j >= n:
            break
        start_t += window_s - overlap_s


def snap_clip(segments, i0, i1, duration):
    i0 = max(0, min(i0, len(segments) - 1))
    i1 = max(i0, min(i1, len(segments) - 1))
    start = float(segments[i0]["start"])
    end = float(segments[i1]["end"])
    end = min(end, duration)
    text = " ".join(segments[k]["text"] for k in range(i0, i1 + 1)).strip()
    return start, end, text


def dedupe(clips, min_gap):
    """Drop near-duplicate clips (overlapping start times), keep higher score."""
    clips = sorted(clips, key=lambda c: (-c["score"], c["start"]))
    kept = []
    for c in clips:
        if any(abs(c["start"] - k["start"]) < min_gap and
               abs(c["end"] - k["end"]) < min_gap for k in kept):
            continue
        # also skip heavy time overlap with an already-kept, higher-scored clip
        if any(not (c["end"] <= k["start"] or c["start"] >= k["end"]) and
               (min(c["end"], k["end"]) - max(c["start"], k["start"])) >
               0.5 * (c["end"] - c["start"]) for k in kept):
            continue
        kept.append(c)
    return kept


def write_markdown(md_path, media, clips):
    lines = [f"# 金句片段候选 — {Path(media).name}", "",
             f"共 {len(clips)} 个候选，按可传播度排序。勾选你想剪的编号后进入 Stage 3。", ""]
    for c in clips:
        lines += [
            f"## #{c['id']}  「{c['hook']}」",
            f"- **时间**：{c['start_ts']} – {c['end_ts']}  （{c['duration']:.0f} 秒）",
            f"- **评分**：{c['score']}/100",
            f"- **理由**：{c['reason']}",
            "",
            "**完整对话：**", "", f"> {c['transcript']}", "",
        ]
    Path(md_path).write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 2: find golden-quote clips")
    p.add_argument("transcript", help="*_highlights_transcript.json from Stage 1")
    p.add_argument("-o", "--output", help="output *_highlights.json path")
    p.add_argument("--backend", choices=["openrouter", "local"], default="openrouter")
    p.add_argument("--base-url", help="override chat base URL")
    p.add_argument("--model", help="override chat model")
    p.add_argument("--min-seconds", type=int, default=45)
    p.add_argument("--max-seconds", type=int, default=75)
    p.add_argument("--max-clips", type=int, default=8, help="max clips to keep overall")
    p.add_argument("--window-seconds", type=int, default=1200,
                   help="transcript window per LLM call for long videos")
    p.add_argument("--overlap-seconds", type=int, default=60)
    args = p.parse_args()

    tpath = Path(args.transcript)
    if not tpath.exists():
        p.error(f"transcript not found: {tpath}")
    data = read_json(tpath)
    segments = data.get("segments", [])
    duration = float(data.get("duration") or (segments[-1]["end"] if segments else 0))
    if not segments:
        p.error("transcript has no segments")

    repo_root = Path(__file__).resolve().parents[4]
    load_env_file(repo_root)
    base, default_model, key_env = BACKENDS[args.backend]
    base_url = args.base_url or base
    model = args.model or default_model
    api_key = os.environ.get(key_env, "" if args.backend == "local" else None)
    if api_key is None:
        p.error(f"{key_env} not set (needed for backend '{args.backend}')")

    per_window = max(3, args.max_clips)
    raw = []
    failed_windows = 0
    win_list = list(windows(segments, args.window_seconds, args.overlap_seconds))
    for wi, (i0, i1) in enumerate(win_list):
        log(f"window {wi + 1}/{len(win_list)} segments [{i0}:{i1}]")
        prompt = PROMPT_TEMPLATE.format(
            min_s=args.min_seconds, max_s=args.max_seconds,
            max_clips=per_window, lines=build_lines(segments, i0, i1))
        try:
            result = chat_json(base_url, api_key or "sk-none", model, prompt)
        except RuntimeError as exc:
            # One bad window must not discard the whole scan of a multi-hour video;
            # windows overlap, so a neighbour usually covers the same ground.
            log(f"window {wi + 1} failed, skipping: {exc}")
            failed_windows += 1
            continue
        for c in result.get("clips", []) or []:
            try:
                raw.append({"start_index": int(c["start_index"]),
                            "end_index": int(c["end_index"]),
                            "hook": str(c.get("hook", "")).strip(),
                            "score": int(c.get("score", 0)),
                            "reason": str(c.get("reason", "")).strip()})
            except (KeyError, ValueError, TypeError):
                continue

    clips = []
    for c in raw:
        start, end, text = snap_clip(segments, c["start_index"], c["end_index"], duration)
        dur = end - start
        if dur < args.min_seconds * 0.6 or dur > args.max_seconds * 1.8:
            continue
        clips.append({"hook": c["hook"], "score": c["score"], "reason": c["reason"],
                      "start": round(start, 2), "end": round(end, 2),
                      "duration": round(dur, 2), "transcript": text})

    if failed_windows == len(win_list):
        raise SystemExit("every window failed; no highlights to report")
    clips = dedupe(clips, min_gap=5.0)[:args.max_clips]
    for idx, c in enumerate(clips, 1):
        c["id"] = idx
        c["start_ts"] = stamp(c["start"])
        c["end_ts"] = stamp(c["end"])

    out_path = Path(args.output) if args.output else \
        tpath.with_name(tpath.name.replace("_highlights_transcript.json",
                                           "_highlights.json"))
    if out_path == tpath:
        out_path = tpath.with_name(tpath.stem + "_highlights.json")
    write_json(out_path, {"media": data.get("media"), "duration": duration,
                          "source_transcript": str(tpath), "backend": args.backend,
                          "model": model, "clips": clips})
    md_path = out_path.with_suffix(".md")
    write_markdown(md_path, data.get("media", ""), clips)
    if failed_windows:
        log(f"warning: {failed_windows}/{len(win_list)} windows were skipped")
    log(f"kept {len(clips)} clips -> {out_path}")
    log(f"report -> {md_path}")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
