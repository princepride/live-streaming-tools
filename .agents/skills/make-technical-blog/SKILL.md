---
name: make-technical-blog
description: Turn a technical video or audio recording plus a PDF or PowerPoint slide deck into a polished illustrated technical blog using this repository's pipeline. Use when generating or regenerating Chinese and English Markdown, DOCX, and PDF editions; reusing chapter transcripts; applying the configured technical writing workflow; or packaging slide images with publishable blog artifacts.
---

# Generate a Technical Blog

Work from the repository root containing `tech_blog_pipeline.py`.

## Prepare inputs

1. Confirm the video or audio file and slide deck exist.
2. Accept PDF, PPTX, PPSX, POTX, PPTM, PPSM, or POTM slides. Let the pipeline render PowerPoint inputs automatically.
3. Look for a matching `*_chapters_transcript.json` and reuse it with `--transcript-json` to avoid duplicate transcription.
4. Read the API key only from `OPENROUTER_API_KEY`. Never print, save, or commit it.
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
  "C:\path\slides.pptx" `
  --transcript-json "D:\path\video_chapters_transcript.json" `
  -o "tech_blog_output\topic_slug" `
  --workers 3 `
  --max-sections 8
```

Omit `--transcript-json` when no reusable transcript exists. Use `--no-english` only when the user requests Chinese-only output. Keep `--max-sections` between 6 and 9.

Allow the pipeline to execute slide analysis, evidence extraction, causal outlining, parallel drafting, global editing, independent review, repair, translation, asset packaging, and deterministic QA. Resume the same command after interruption so cache keys remain useful.

## Verify the deliverables

Run:

```powershell
python .agents\skills\validate-stream-artifacts\scripts\validate_artifacts.py blog `
  "tech_blog_output\topic_slug" --require-english
```

Then visually inspect every PDF page. Render both DOCX files with the document workflow when Word or LibreOffice is available; otherwise disclose that visual DOCX verification was unavailable and report the structural audit separately.

Require the final directory to contain Chinese Markdown, DOCX, and PDF; English equivalents unless disabled; packaged `assets/slides/`; and a passing `qa-report.json`. Ensure Markdown image links are relative and resolve inside the final directory.

## Update the project index

When the blog corresponds to a README index entry, place the project folder name in the “博客” column and link it directly to `./tech_blog_output/<topic_slug>/final/blog.md`. Leave entries without a blog blank.

Report direct paths to both Markdown editions and each DOCX/PDF deliverable, plus deterministic and visual QA results.
