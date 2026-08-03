---
name: validate-stream-artifacts
description: Validate outputs produced by this repository's chapter and technical-blog pipelines. Use when checking chapter JSON constraints and timeline coverage, Markdown structure and local images, bilingual parity, DOCX package integrity and table geometry, PDF pages and embedded images, or final qa-report status before committing or publishing generated artifacts.
---

# Validate Stream Tools Artifacts

Use the bundled deterministic validator first, then perform visual checks for rendered documents.

## Validate chapter JSON

Run from the repository root:

```powershell
python .agents\skills\validate-stream-artifacts\scripts\validate_artifacts.py chapters `
  "path\to\video_chapters.json" `
  --max-chapters 10 `
  --max-title-chars 16
```

Treat any count, title-length, ordering, gap, overlap, duration, or full-coverage failure as blocking.

## Validate a blog output

Pass either the project output directory or its `final` directory:

```powershell
python .agents\skills\validate-stream-artifacts\scripts\validate_artifacts.py blog `
  "tech_blog_output\topic_slug" `
  --require-english `
  --json-output "tmp\topic_slug-validation.json"
```

The script checks required files, Markdown heading progression, image existence and path safety, Chinese/English image parity, `qa-report.json`, DOCX integrity, and PDF structure. Fix all deterministic failures before visual review.

## Perform visual QA

1. Render every page of each final PDF and inspect for clipping, overlap, blank pages, missing glyphs, malformed tables, unreadable images, and excessive layout gaps.
2. Render each final DOCX with the document rendering workflow and inspect every page.
3. If the DOCX renderer is unavailable, do not call it visually verified. Record the limitation and retain the passing ZIP, media, accessibility, and table-geometry audits.
4. Re-render all affected pages after any source or renderer fix.

## Report results

Separate deterministic results from visual results. Include exact failing checks, affected paths or pages, and the final overall status. Do not expose API keys or include ignored caches and transcripts in publication recommendations.

The validator exits with code `0` on success and `1` on validation failure.
