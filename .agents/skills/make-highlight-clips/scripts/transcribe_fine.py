"""Stage 1: fine-timestamp transcription for highlight clipping.

Thin wrapper over the repo's shared stt.py (mode="fine"), producing
sentence/segment-level (and optional word-level) timestamps accurate enough to
cut ~60s clips. Default backend openrouter (openai/whisper-large-v3), which
returns verbose_json timestamps and reuses OPENROUTER_API_KEY.

Output: <stem>_highlights_transcript.json
{media, duration, language, backend, model, segments:[{start,end,text,words?}]}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_env_file, log, write_json  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
import stt  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(
        description="Stage 1: fine-timestamp transcription (shared stt.py)")
    p.add_argument("media", help="path to video/audio file")
    p.add_argument("-o", "--output", help="output transcript JSON path")
    p.add_argument("--backend", choices=list(stt.BACKENDS), default="openrouter")
    p.add_argument("--base-url", help="override OpenAI-compatible base URL")
    p.add_argument("--model", help="override ASR model")
    p.add_argument("--language", default="zh", help="ISO language code (default zh)")
    p.add_argument("--chunk-seconds", type=int, default=480,
                   help="API audio chunk length to stay under upload limits")
    p.add_argument("--words", action="store_true",
                   help="also request word-level timestamps")
    args = p.parse_args()

    media = Path(args.media).expanduser()
    if not media.exists():
        p.error(f"media not found: {media}")

    load_env_file(REPO_ROOT)
    try:
        segments = stt.transcribe(
            media, backend=args.backend, base_url=args.base_url, model=args.model,
            language=args.language, chunk_seconds=args.chunk_seconds, mode="fine",
            want_words=args.words)
    except RuntimeError as exc:
        p.error(str(exc))

    _, model, _ = stt.resolve(args.backend, args.base_url, args.model)
    duration = stt.media_duration(media)
    out_path = Path(args.output) if args.output else \
        media.with_name(f"{media.stem}_highlights_transcript.json")
    write_json(out_path, {
        "media": str(media), "duration": round(duration, 3),
        "language": args.language, "backend": args.backend, "model": model,
        "segments": segments,
    })
    log(f"wrote {len(segments)} segments -> {out_path}")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
