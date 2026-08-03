---
name: make-video-chapters
description: Generate Bilibili-ready chapters from a local video or audio file, or download and process an entire Bilibili collection with BBDown. Use for media transcription, timestamped chapter planning, collection audio downloads, cached reruns, or enforcing this project's limits of at most 10 chapters and at most 16 characters per title.
---

# Generate Video Chapters

Work from the repository root containing `auto_chapters.py`.

## Choose the workflow

- For one local video or audio file, run `auto_chapters.py`.
- For a Bilibili collection, run `batch_bilibili_chapters.py`.
- Reuse an existing `*_chapters_transcript.json` when a later blog task needs the same transcript.

## Prepare the environment

1. Confirm the media file exists.
2. Confirm `ffmpeg` and `ffprobe` are available.
3. Read the OpenRouter key only from `OPENROUTER_API_KEY`. Never print, persist, or commit it.
4. For collection downloads, confirm the exact `BBDown.exe` path.

## Process one media file

Run:

```powershell
python auto_chapters.py "D:\path\video.mp4" `
  --max-chapters 10 `
  --max-title-chars 16 `
  --min-minutes 6 `
  --max-minutes 10
```

Use `-o` for an explicit JSON path. Preserve the 10-chapter and 16-character limits unless the user explicitly changes them. Prefer the default high-quality chapter model. Re-run the same command to reuse transcription caches after interruption.

Expected outputs are `*_chapters.json`, `*_chapters.txt`, and `*_chapters_transcript.json`.

## Process a Bilibili collection

Run:

```powershell
python batch_bilibili_chapters.py `
  --mid "<up主 mid>" `
  --sid "<合集 sid>" `
  --bbdown "C:\path\BBDown.exe" `
  --output-dir "D:\path\bilibili_audio" `
  --max-chapters 10 `
  --max-title-chars 16
```

Use `--download-only`, `--chapters-only`, or `--force-chapters` only when the requested workflow calls for them. Inspect `failures.json`; retry failed items without discarding successful downloads or caches.

## Validate and report

Run the project validator after generation:

```powershell
python .agents\skills\validate-stream-artifacts\scripts\validate_artifacts.py chapters "path\to\chapters.json"
```

Require full-duration coverage, contiguous non-overlapping ranges, positive durations, no more than 10 chapters, and titles of at most 16 characters. Report the output paths, chapter count, model used, and any failed collection items.
