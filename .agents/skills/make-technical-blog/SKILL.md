---
name: make-technical-blog
description: Turn a technical video or audio recording plus a PDF or PowerPoint slide deck into a polished illustrated technical blog using this repository's pipeline. Use when generating or regenerating Chinese and English Markdown, DOCX, and PDF editions; reusing chapter transcripts; applying the configured technical writing workflow; or packaging slide images with publishable blog artifacts.
---

# Generate a Technical Blog

Work from the repository root containing `tech_blog_pipeline.py`.

## Prepare inputs

1. Confirm the video or audio file and slide deck exist.
2. Accept one or more PDF, PPTX, PPSX, POTX, PPTM, PPSM, or POTM slide decks. Pass multiple decks as consecutive positional arguments after the media file; the pipeline combines them into one ordered, globally numbered evidence set while retaining each source filename and original page number. Let the pipeline render PowerPoint inputs automatically.
   - When every PowerPoint page is a single full-slide raster image, preserve the original embedded image instead of rasterizing it again. The pipeline will focus-crop decorative whitespace for the publishable blog asset while retaining the untouched source in its cache.
   - For other PowerPoint decks, render at the pipeline's high-resolution setting. Never upscale a previously downsampled blog asset when the source deck is available.
3. Look for a matching `*_chapters_transcript.json` and reuse it with `--transcript-json` to avoid duplicate transcription.
4. Choose `--provider openrouter` for the OpenRouter API or `--provider codex` for an authenticated Codex CLI. For OpenRouter, use `OPENROUTER_API_KEY` from the process environment or repository-root `.env`. For Codex, require an existing transcript JSON and never send the audio file to the model backend.
5. Choose a unique, descriptive output folder under `tech_blog_output/`.

## Run the quality workflow

Use the configured high-quality defaults unless the user specifies models:

- Analysis and review: `google/gemini-3.1-pro-preview`
- Writing and English translation: `anthropic/claude-opus-4.6`
- Transcription: `openai/gpt-4o-mini-transcribe`

Run:

```powershell
python tech_blog_pipeline.py `
  "D:\path\video.mp4" `
  "C:\path\slides-part-1.pptx" `
  "C:\path\slides-part-2.pptx" `
  --transcript-json "D:\path\video_chapters_transcript.json" `
  -o "tech_blog_output\topic_slug" `
  --workers 3 `
  --max-sections 8
```

The default is `--provider openrouter`. To use Codex instead, authenticate the CLI with `codex login`, reuse a transcript, and run:

```powershell
python tech_blog_pipeline.py `
  "D:\path\video.mp4" `
  "C:\path\slides.pptx" `
  --transcript-json "D:\path\video_chapters_transcript.json" `
  --provider codex `
  -o "tech_blog_output\topic_slug" `
  --workers 3 `
  --max-sections 8
```

Use `--codex-model <model>` only when the user requests a specific Codex model; otherwise retain the authenticated CLI's configured default. Codex mode may send transcript text and rendered slide images to OpenAI, but not the source audio. OpenRouter mode may send audio when no reusable transcript exists.

Omit `--transcript-json` only with the OpenRouter backend. Use `--no-english` only when the user requests Chinese-only output. Keep `--max-sections` between 6 and 9.

Allow the pipeline to execute slide analysis, evidence extraction, causal outlining, parallel drafting, global editing, independent review, repair, translation, asset packaging, and deterministic QA. Resume the same command after interruption so cache keys remain useful.

When a central tensor formula, attention path, sharding layout, or collective operation remains unclear in the source slides, use `$tensor-formula-viz` to create one supplementary shape-aware figure. Add it only when it improves the explanation, store its publishable files under `final/assets/diagrams/`, and reference the PNG or SVG with a relative Markdown path. Keep the editable/vector source beside it and do not invent shapes that are not supported by the evidence ledger.

## Verify the deliverables

Run:

```powershell
python .agents\skills\validate-stream-artifacts\scripts\validate_artifacts.py blog `
  "tech_blog_output\topic_slug" --require-english
```

Then visually inspect every PDF page. Render both DOCX files with the document workflow when Word or LibreOffice is available; otherwise disclose that visual DOCX verification was unavailable and report the structural audit separately.

During visual inspection, reject slide images whose useful chart or diagram occupies only a small fraction of the frame. Prefer a lossless crop around the evidence region, leaving enough padding for labels and legends. Confirm the cropped asset is sharper at publication width and does not remove lightly colored arrows, annotations, axes, or table cells.

Require the final directory to contain Chinese Markdown, DOCX, and PDF; English equivalents unless disabled; packaged `assets/slides/`; and a passing `qa-report.json`. Ensure Markdown image links are relative and resolve inside the final directory.

## Update the project index

When the blog corresponds to a README index entry, place the project folder name in the “博客” column and link it directly to `./tech_blog_output/<topic_slug>/final/blog.md`. Leave entries without a blog blank.

Report direct paths to both Markdown editions and each DOCX/PDF deliverable, plus deterministic and visual QA results.
