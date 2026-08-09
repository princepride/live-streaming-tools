---
name: make-highlight-clips
description: Find the golden-quote ("金句") and highlight moments in a long video and cut them into short ~60-second clips. Use whenever the user wants to mine a long talk, livestream, or interview for shareable highlights, quotable moments, or short vertical clips; wants a timestamped highlight report to choose from before cutting; or asks to auto-transcribe, detect, review, and export short clips. Runs a four-stage human-in-the-loop pipeline (transcribe → find highlights → review → cut) and supports both OpenRouter and a local model server for the highlight-detection step.
---

# Make Highlight Clips

Mine a long video for its best short moments and cut them, with a human-in-the-loop
review step. Work from the repository root (the folder containing `auto_chapters.py`).

The pipeline has four stages. Each stage writes a file the next stage reads, so you
can stop, let the user decide, and resume. **Do not run all four stages silently in
one shot** — Stage 3 exists so the user chooses which clips get cut.

```
video.mp4
  │  Stage 1  transcribe_fine.py   → <stem>_highlights_transcript.json
  │  Stage 2  find_highlights.py   → <stem>_highlights.json  +  <stem>_highlights.md
  │  Stage 3  build_review.py      → <stem>_highlights_review.html  (user picks)
  │           user exports         → <stem>_selection.json
  │  Stage 4  cut_clips.py         → clips/<stem>_NN_landscape.mp4 + _vertical.mp4
```

Scripts live in `.agents/skills/make-highlight-clips/scripts/`. Run them with the
repo root as the working directory.

## Prepare the environment

1. Confirm the media file, `ffmpeg`, and `ffprobe` are available.
2. API keys are read only from environment variables, falling back to the
   git-ignored `.env` at the repo root. Never print, persist, or commit them.
   - Stage 1 (ASR): `OPENROUTER_API_KEY` (default) — the same key used elsewhere in
     this repo; or `GROQ_API_KEY` / `OPENAI_API_KEY` for those providers directly.
   - Stage 2 (highlight LLM): `OPENROUTER_API_KEY` (default) or a local server.
3. `pip install requests` (already in `requirements.txt`). For `--backend local`
   in Stage 1, also `pip install faster-whisper`.

## Stage 1 — Fine-timestamp transcription

This stage produces real segment-level (and, with `--words`, word-level)
timestamps. It shares the repo's transcription core `stt.py` with the chapter
pipeline (`auto_chapters.py`); both run it in `fine` mode, so chapters and clips
alike are cut on accurate timestamps.

OpenRouter's `/audio/transcriptions` endpoint returns segment timestamps when
`response_format=verbose_json` is routed to an OpenAI-compatible provider; the
default model `openai/whisper-large-v3` is hosted by Groq/Together and qualifies, so
Stage 1 can run on the same `OPENROUTER_API_KEY` as the rest of the pipeline.

```powershell
python .agents\skills\make-highlight-clips\scripts\transcribe_fine.py "D:\path\video.mp4" `
  --backend openrouter --language zh
```

- `--backend openrouter` (default, `OPENROUTER_API_KEY`, model
  `openai/whisper-large-v3`) — verbose_json segment timestamps, one key for the
  whole pipeline.
- `--backend groq` (`GROQ_API_KEY`, `whisper-large-v3`) — direct, fast, cheap.
- `--backend openai` (`OPENAI_API_KEY`, `whisper-1`) — verbose_json timestamps.
- `--backend local` — offline `faster-whisper`, word-level timestamps, no key.
- `--base-url` / `--model` override any OpenAI-compatible ASR endpoint.
- `--chunk-seconds` (default 480) splits long audio to respect the 25 MB upload cap
  and 60s upstream timeout; timestamps are re-offset per chunk so they stay
  absolute. Lower it if you hit provider timeouts.

Output `<stem>_highlights_transcript.json`:
`{media, duration, language, backend, model, segments:[{start, end, text, words?}]}`.
If a valid transcript already exists, reuse it instead of re-transcribing.

## Stage 2 — Find golden-quote clips

`find_highlights.py` scans the transcript with an LLM and proposes multiple
self-contained ~60s candidate clips. Long transcripts are processed in overlapping
time windows and merged, so multi-hour videos work.

```powershell
python .agents\skills\make-highlight-clips\scripts\find_highlights.py `
  "D:\path\video_highlights_transcript.json" `
  --backend openrouter --min-seconds 45 --max-seconds 75 --max-clips 8
```

- `--backend openrouter` (default, `OPENROUTER_API_KEY`, model
  `google/gemini-3.1-pro-preview`) — matches this repo's chapter pipeline.
- `--backend local` — any OpenAI-compatible server; set `--base-url`
  (default `http://localhost:8000/v1`) and `--model`, key from `LOCAL_API_KEY`.
- `--min-seconds` / `--max-seconds` clip length band (default 45–75s).
- `--max-clips` overall cap (default 8), ranked by a 0–100 quotability score.
- `--window-seconds` / `--overlap-seconds` control long-video windowing.

The model returns segment index ranges; clips are **snapped to transcript segment
boundaries** so they never start or end mid-sentence, then de-duplicated by time
overlap. Each kept clip has: `id, hook (金句), start/end, start_ts/end_ts, duration,
score, reason, transcript` (full dialogue).

Outputs `<stem>_highlights.json` and a human-readable `<stem>_highlights.md`.
Show the Markdown report to the user before Stage 3.

## Stage 3 — Review and pick

`build_review.py` renders a **self-contained** interactive HTML page (no server, no
network) of checkbox cards — each shows the hook, timestamps, score, reason, and the
full dialogue in a collapsible block.

```powershell
python .agents\skills\make-highlight-clips\scripts\build_review.py `
  "D:\path\video_highlights.json"
```

Open `<stem>_highlights_review.html` in a browser. Tick the clips to keep, then
either click **导出所选** to download `<stem>_selection.json`, or **复制 --pick** to
copy a `--pick 1,3,5` string for the CLI. Wait for the user's choice before Stage 4.

## Stage 4 — Cut the clips

`cut_clips.py` reads the selection and cuts each pick with ffmpeg (accurate input
seek + re-encode). Every clip is exported twice: a **landscape** original-resolution
mp4 and a **vertical 9:16** (1080×1920) mp4 with a blurred-fill background.

```powershell
# from the review page's selection.json:
python .agents\skills\make-highlight-clips\scripts\cut_clips.py `
  "D:\path\video_selection.json" --burn-subs

# or straight from highlights.json with explicit picks:
python .agents\skills\make-highlight-clips\scripts\cut_clips.py `
  "D:\path\video_highlights.json" --pick 1,3,5 --media "D:\path\video.mp4"
```

- `--burn-subs` burns sentence subtitles built from the Stage 1 transcript
  (`--transcript` to point at it explicitly; otherwise read from the JSON).
- `--no-landscape` / `--no-vertical` to export only one orientation.
- `--outdir` (default `<media_dir>/clips`).

Outputs `clips/<stem>_NN_landscape.mp4` and `clips/<stem>_NN_vertical.mp4`.

## Full run (summary)

```powershell
python .agents\skills\make-highlight-clips\scripts\transcribe_fine.py "video.mp4"
python .agents\skills\make-highlight-clips\scripts\find_highlights.py "video_highlights_transcript.json"
python .agents\skills\make-highlight-clips\scripts\build_review.py "video_highlights.json"
# → user opens the HTML, picks clips, downloads video_selection.json
python .agents\skills\make-highlight-clips\scripts\cut_clips.py "video_selection.json" --burn-subs
```

Report the transcript path, the number of candidates and the Markdown report path,
then the final clip files after the user picks.
