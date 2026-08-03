#!/usr/bin/env python3
"""Deterministically validate chapter and blog artifacts from this repository."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+\S", re.MULTILINE)


def find_repo(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "blog_docx.py").is_file() and (candidate / "auto_chapters.py").is_file():
            return candidate
    raise RuntimeError("Cannot locate repository root containing blog_docx.py and auto_chapters.py")


def add(checks: dict[str, bool], issues: list[str], name: str, passed: bool, message: str) -> None:
    checks[name] = bool(passed)
    if not passed:
        issues.append(message)


def validate_chapters(path: Path, max_chapters: int, max_title_chars: int) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    chapters = data.get("chapters")
    duration = data.get("duration")
    checks: dict[str, bool] = {}
    issues: list[str] = []

    add(checks, issues, "chapters_is_list", isinstance(chapters, list) and bool(chapters),
        "chapters must be a non-empty list")
    if not isinstance(chapters, list) or not chapters:
        return {"kind": "chapters", "path": str(path.resolve()), "pass": False,
                "checks": checks, "issues": issues}

    add(checks, issues, "chapter_count", len(chapters) <= max_chapters,
        f"chapter count {len(chapters)} exceeds {max_chapters}")
    titles = [str(item.get("title", "")).strip() for item in chapters]
    long_titles = [(index + 1, title, len(title)) for index, title in enumerate(titles)
                   if len(title) > max_title_chars]
    add(checks, issues, "title_lengths", not long_titles,
        f"titles exceed {max_title_chars} characters: {long_titles}")
    add(checks, issues, "nonempty_titles", all(titles), "one or more chapter titles are empty")
    add(checks, issues, "summaries", all(str(item.get("summary", "")).strip() for item in chapters),
        "one or more chapter summaries are empty")

    ranges: list[tuple[float, float]] = []
    valid_ranges = True
    for item in chapters:
        try:
            start, end = float(item["start"]), float(item["end"])
            ranges.append((start, end))
            valid_ranges = valid_ranges and start >= 0 and end > start
        except (KeyError, TypeError, ValueError):
            valid_ranges = False
    add(checks, issues, "valid_ranges", valid_ranges and len(ranges) == len(chapters),
        "chapter ranges must contain numeric start/end values with end > start >= 0")

    if valid_ranges and len(ranges) == len(chapters):
        contiguous = all(abs(left[1] - right[0]) <= 1 for left, right in zip(ranges, ranges[1:]))
        add(checks, issues, "contiguous", contiguous, "chapter ranges contain a gap or overlap")
        add(checks, issues, "starts_at_zero", abs(ranges[0][0]) <= 1,
            f"first chapter starts at {ranges[0][0]}, not 0")
        numeric_duration = isinstance(duration, (int, float)) and duration > 0
        add(checks, issues, "duration_present", numeric_duration, "duration must be a positive number")
        if numeric_duration:
            add(checks, issues, "full_coverage", abs(ranges[-1][1] - float(duration)) <= 1,
                f"last chapter ends at {ranges[-1][1]}, media duration is {duration}")

    return {
        "kind": "chapters", "path": str(path.resolve()), "pass": not issues,
        "chapter_count": len(chapters), "checks": checks, "issues": issues,
    }


def markdown_audit(path: Path, final_dir: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    images = IMAGE_RE.findall(text)
    missing: list[str] = []
    unsafe: list[str] = []
    for raw in images:
        target = raw.split("#", 1)[0].strip()
        if re.match(r"^[a-z]+://", target, re.I):
            continue
        resolved = (path.parent / target).resolve()
        try:
            resolved.relative_to(final_dir.resolve())
        except ValueError:
            unsafe.append(raw)
        if not resolved.is_file():
            missing.append(raw)
    levels = [len(match.group(1)) for match in HEADING_RE.finditer(text)]
    jumps = [(index + 1, left, right) for index, (left, right) in
             enumerate(zip(levels, levels[1:])) if right > left + 1]
    return {
        "path": str(path.resolve()), "pass": bool(text.strip()) and not missing and not unsafe and not jumps,
        "characters": len(text), "image_paths": images, "missing_images": missing,
        "unsafe_image_paths": unsafe, "heading_jumps": jumps,
    }


def validate_blog(path: Path, require_english: bool) -> dict[str, Any]:
    final_dir = (path / "final") if (path / "final").is_dir() else path
    repo = find_repo(Path(__file__))
    sys.path.insert(0, str(repo))
    from blog_docx import audit_docx  # pylint: disable=import-outside-toplevel
    from blog_pdf import audit_pdf  # pylint: disable=import-outside-toplevel

    checks: dict[str, bool] = {}
    issues: list[str] = []
    required = ["blog.md", "blog.docx", "blog.pdf"]
    english_names = ["blog.en.md", "blog.en.docx", "blog.en.pdf"]
    english_present = any((final_dir / name).exists() for name in english_names)
    if require_english or english_present:
        required.extend(english_names)
    missing_files = [name for name in required if not (final_dir / name).is_file()]
    add(checks, issues, "required_files", not missing_files, f"missing required files: {missing_files}")

    result: dict[str, Any] = {
        "kind": "blog", "path": str(final_dir.resolve()), "checks": checks, "issues": issues,
    }
    if missing_files:
        result["pass"] = False
        return result

    zh_md = markdown_audit(final_dir / "blog.md", final_dir)
    result["markdown"] = zh_md
    add(checks, issues, "chinese_markdown", zh_md["pass"], "Chinese Markdown audit failed")
    expected_images = len(zh_md["image_paths"])
    zh_docx = audit_docx(final_dir / "blog.docx")
    zh_pdf = audit_pdf(final_dir / "blog.pdf", expected_images=expected_images)
    result["docx"] = zh_docx
    result["pdf"] = zh_pdf
    add(checks, issues, "chinese_docx", bool(zh_docx.get("pass")), "Chinese DOCX audit failed")
    add(checks, issues, "chinese_pdf", bool(zh_pdf.get("pass")), "Chinese PDF audit failed")

    if require_english or english_present:
        en_md = markdown_audit(final_dir / "blog.en.md", final_dir)
        en_docx = audit_docx(final_dir / "blog.en.docx")
        en_pdf = audit_pdf(final_dir / "blog.en.pdf", expected_images=expected_images)
        parity = zh_md["image_paths"] == en_md["image_paths"]
        result["english"] = {"markdown": en_md, "docx": en_docx, "pdf": en_pdf,
                             "image_path_parity": parity}
        add(checks, issues, "english_markdown", en_md["pass"], "English Markdown audit failed")
        add(checks, issues, "english_docx", bool(en_docx.get("pass")), "English DOCX audit failed")
        add(checks, issues, "english_pdf", bool(en_pdf.get("pass")), "English PDF audit failed")
        add(checks, issues, "bilingual_image_parity", parity,
            "Chinese and English Markdown image paths differ")

    qa_path = final_dir / "qa-report.json"
    if qa_path.is_file():
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        result["qa_report"] = {"path": str(qa_path.resolve()), "pass": qa.get("pass") is True}
        add(checks, issues, "qa_report", qa.get("pass") is True, "qa-report.json is not passing")
    else:
        result["qa_report"] = {"path": str(qa_path.resolve()), "present": False}
        add(checks, issues, "qa_report", False, "qa-report.json is missing")

    result["pass"] = not issues
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    chapters = subparsers.add_parser("chapters", help="validate a chapter JSON file")
    chapters.add_argument("path", type=Path)
    chapters.add_argument("--max-chapters", type=int, default=10)
    chapters.add_argument("--max-title-chars", type=int, default=16)
    chapters.add_argument("--json-output", type=Path, help="also write the result as JSON")
    blog = subparsers.add_parser("blog", help="validate a blog output directory")
    blog.add_argument("path", type=Path)
    blog.add_argument("--require-english", action="store_true")
    blog.add_argument("--json-output", type=Path, help="also write the result as JSON")
    args = parser.parse_args()

    if not args.path.exists():
        parser.error(f"path does not exist: {args.path}")
    if args.command == "chapters":
        result = validate_chapters(args.path, args.max_chapters, args.max_title_chars)
    else:
        result = validate_blog(args.path, args.require_english)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(payload + "\n", encoding="utf-8")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
