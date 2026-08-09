#!/usr/bin/env python3
"""Download a Bilibili collection's audio with BBDown and chapter every item."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import requests


DEFAULT_BBDOWN = Path(r"C:\Users\wangz\Downloads\BBDown_1.6.3_20240814_win-x64\BBDown.exe")
MEDIA_EXTENSIONS = {".m4a", ".mp3", ".aac", ".flac", ".wav", ".opus", ".ogg", ".mp4"}


def log(message: str) -> None:
    print(message, flush=True)


def get_collection(mid: str, sid: str) -> tuple[dict, list[dict]]:
    url = "https://api.bilibili.com/x/polymer/web-space/seasons_archives_list"
    headers = {
        "Referer": f"https://space.bilibili.com/{mid}/lists?sid={sid}",
        "User-Agent": "Mozilla/5.0",
    }
    items: list[dict] = []
    page = 1
    meta: dict = {}
    while True:
        response = requests.get(url, headers=headers, params={
            "mid": mid, "season_id": sid, "page_num": page, "page_size": 100,
        }, timeout=60)
        response.raise_for_status()
        body = response.json()
        if body.get("code") != 0:
            raise RuntimeError(f"B站合集接口错误：{body.get('message')}")
        data = body["data"]
        meta = data.get("meta") or meta
        items.extend(data.get("archives", []))
        total = int(data["page"]["total"])
        if len(items) >= total:
            return meta, items
        page += 1


def find_audio(folder: Path, bvid: str) -> Path | None:
    matches = [
        path for path in folder.iterdir()
        if path.is_file() and path.stat().st_size > 1024
        and f"[{bvid}]" in path.name and path.suffix.lower() in MEDIA_EXTENSIONS
    ]
    return max(matches, key=lambda path: path.stat().st_size) if matches else None


def download_audio(bbdown: Path, folder: Path, item: dict) -> Path:
    bvid = item["bvid"]
    existing = find_audio(folder, bvid)
    if existing:
        log(f"[跳过下载] {existing.name}")
        return existing
    command = [
        str(bbdown), bvid, "--audio-only", "--hide-streams", "--skip-cover",
        "--skip-subtitle", "--save-archives-to-file", "--work-dir", str(folder),
        "-F", f"[{bvid}]<videoTitle>",
    ]
    subprocess.run(command, check=True)
    downloaded = find_audio(folder, bvid)
    if not downloaded:
        raise RuntimeError(f"BBDown 完成后未找到 {bvid} 的音频文件")
    return downloaded


def make_chapters(media: Path, output_dir: Path, args: argparse.Namespace) -> None:
    output = output_dir / f"{media.stem}_chapters.json"
    text_output = output.with_suffix(".txt")
    if output.is_file() and text_output.is_file() and not args.force_chapters:
        log(f"[跳过分章] {text_output.name}")
        return
    command = [
        sys.executable, str(Path(__file__).with_name("auto_chapters.py")), str(media),
        "-o", str(output), "--max-chapters", str(args.max_chapters),
        "--max-title-chars", str(args.max_title_chars), "--workers", str(args.workers),
        "--chapter-model", args.chapter_model,
        "--stt-backend", args.stt_backend, "--stt-model", args.stt_model,
    ]
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="批量下载B站合集音频并自动分章节")
    parser.add_argument("--mid", default="189708420", help="UP主 mid")
    parser.add_argument("--sid", default="8336139", help="合集 sid")
    parser.add_argument("--bbdown", type=Path, default=DEFAULT_BBDOWN)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "bilibili_audio")
    parser.add_argument("--max-chapters", type=int, default=10)
    parser.add_argument("--max-title-chars", type=int, default=16)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--chapter-model", default="google/gemini-3.1-pro-preview")
    parser.add_argument("--stt-backend", default="openrouter",
                        choices=["openrouter", "groq", "openai", "local"],
                        help="转写后端（默认 openrouter，透传给 auto_chapters.py）")
    parser.add_argument("--stt-model", default="openai/whisper-large-v3",
                        help="转写模型（默认 openai/whisper-large-v3）")
    parser.add_argument("--force-chapters", action="store_true", help="复用转写缓存，强制重新生成章节")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--chapters-only", action="store_true")
    args = parser.parse_args()
    if not args.bbdown.is_file():
        parser.error(f"找不到 BBDown：{args.bbdown}")
    if args.download_only and args.chapters_only:
        parser.error("--download-only 和 --chapters-only 不能同时使用")
    if not args.download_only and not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("分章节需要环境变量 OPENROUTER_API_KEY")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chapter_dir = args.output_dir / "chapters"
    chapter_dir.mkdir(parents=True, exist_ok=True)
    meta, items = get_collection(args.mid, args.sid)
    manifest = {
        "mid": args.mid, "sid": args.sid, "name": meta.get("name"),
        "count": len(items), "items": [
            {key: item.get(key) for key in ("bvid", "title", "duration")} for item in items
        ],
    }
    (args.output_dir / "collection.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"合集：{manifest['name']}，共 {len(items)} 条")

    failures = []
    for index, item in enumerate(items, 1):
        log(f"\n[{index}/{len(items)}] {item['title']} ({item['bvid']})")
        try:
            audio = find_audio(args.output_dir, item["bvid"])
            if not args.chapters_only:
                audio = download_audio(args.bbdown, args.output_dir, item)
            if not args.download_only:
                if not audio:
                    raise RuntimeError("没有可用于分章节的已下载音频")
                make_chapters(audio, chapter_dir, args)
        except Exception as exc:
            failures.append({"bvid": item["bvid"], "title": item["title"], "error": str(exc)})
            log(f"[失败] {exc}")
    (args.output_dir / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if failures:
        log(f"完成，但有 {len(failures)} 条失败；再次运行会跳过已完成下载并继续。")
        return 1
    log("全部音频下载与分章节完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
