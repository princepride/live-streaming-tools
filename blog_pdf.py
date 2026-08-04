"""Create a polished, self-contained PDF from the pipeline's final Markdown."""

from __future__ import annotations

import html
import re
from datetime import date
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from blog_docx import INLINE_PATTERN, _extract_title, _parse_table, inline_math_to_plain, latex_to_plain

ZERO_WIDTH_SPACE = "​"

# jieba is optional — it greatly improves CJK line-breaking in PDFs but isn't
# required for English output. Load and warm it once at import time.
_jieba_cut = None
try:
    import jieba
    _jieba_cut = jieba.lcut
    # Force dictionary load eagerly so first render isn't slow.
    _ = _jieba_cut("预热")
except ImportError:
    pass


def _cjk_line_break(text: str) -> str:
    """Insert zero-width spaces between Chinese words so ReportLab can break naturally.

    ReportLab breaks lines on characters (including spaces) but has no notion of CJK
    word boundaries.  Without word segmentation every character is an equally valid
    break point, so lines routinely split words in the middle.  Inserting U+200B at
    segmentation boundaries gives the layout engine preferred break positions while
    keeping the rendered text visually identical.
    """
    if _jieba_cut is None:
        return text  # jieba not available — fall back to ReportLab's per-character breaks
    # Segment the full text as a single string so jieba can use contextual cues
    # (short adjacent tokens, English fragments, numbers) to produce better cuts.
    tokens = _jieba_cut(text)
    return ZERO_WIDTH_SPACE.join(tokens)


def _cjk_break_html(html: str) -> str:
    """Apply CJK word segmentation to text portions of an HTML string.

    Only the content outside ``<tag>`` (including inside ``</tag>``) is segmented;
    attribute values inside opening tags are left alone because they have no impact
    on ReportLab line-breaking.  Text inside ``<font name="{MONO_FONT}">`` (inline
    code) is also skipped — segmenting identifiers like ``generate_sequences()``
    would insert zero-width spaces mid-identifier and break the code rendering.
    """
    if _jieba_cut is None:
        return html
    # Split into tagged blocks; skip mono-font segments entirely.
    escape = rf'<font\s+name="{MONO_FONT}"[^>]*>.*?</font>'
    placeholders: dict[str, str] = {}

    def _save(m: re.Match) -> str:
        key = f"{len(placeholders)}"
        placeholders[key] = m.group(0)
        return key

    html = re.sub(escape, _save, html, flags=re.S)
    # Segment the remaining text (outside tags and outside mono blocks).
    html = re.sub(r">([^<]+)<", lambda m: ">" + _cjk_line_break(m.group(1)) + "<", html)
    # Restore mono-font blocks verbatim.
    for key, value in placeholders.items():
        html = html.replace(key, value, 1)
    return html


INK = colors.HexColor("#203748")
BLUE = colors.HexColor("#2E74B5")
DARK_BLUE = colors.HexColor("#1F4D78")
MUTED = colors.HexColor("#626B75")
LIGHT = colors.HexColor("#F4F6F9")
BORDER = colors.HexColor("#CAD2DC")
LINK = colors.HexColor("#0563C1")


def _font_path(*names: str) -> Path:
    roots = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts"), Path("/Library/Fonts")]
    # Many Linux distros nest fonts in subdirectories (truetype/, opentype/, etc.).
    for root in roots:
        if not root.is_dir():
            continue
        for name in names:
            candidates = list(root.rglob(name)) if root.is_dir() else []
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
    raise RuntimeError(f"找不到可用于中文 PDF 的字体：{', '.join(names)}")


MONO_FONT = "BlogMono"


def register_fonts(language: str = "zh") -> tuple[str, str]:
    if language not in {"zh", "en"}:
        raise ValueError("language must be 'zh' or 'en'")
    regular_name = "BlogLatin" if language == "en" else "BlogCJK"
    bold_name = "BlogLatinBold" if language == "en" else "BlogCJKBold"
    if regular_name not in pdfmetrics.getRegisteredFontNames():
        if language == "en":
            regular = _font_path("arial.ttf", "LiberationSans-Regular.ttf")
            bold = _font_path("arialbd.ttf", "LiberationSans-Bold.ttf")
        else:
            regular = _font_path("msyh.ttc", "simhei.ttf", "NotoSansCJK-Regular.ttc")
            bold = _font_path("msyhbd.ttc", "simhei.ttf", "NotoSansCJK-Bold.ttc")
        pdfmetrics.registerFont(TTFont(regular_name, str(regular), subfontIndex=0))
        pdfmetrics.registerFont(TTFont(bold_name, str(bold), subfontIndex=0))
        pdfmetrics.registerFontFamily(regular_name, normal=regular_name, bold=bold_name,
                                      italic=regular_name, boldItalic=bold_name)
    # Register a monospace font for inline code — ReportLab has no built-in Courier.
    if MONO_FONT not in pdfmetrics.getRegisteredFontNames():
        mono_path = _font_path("DejaVuSansMono.ttf", "LiberationMono-Regular.ttf",
                               "DejaVuSansMono.ttc", "LiberationMono-Regular.ttc")
        pdfmetrics.registerFont(TTFont(MONO_FONT, str(mono_path), subfontIndex=0))
    return regular_name, bold_name


def _inline_markup(text: str, *, regular_font: str, bold_font: str) -> str:
    text = inline_math_to_plain(text)
    text = text.translate(str.maketrans({
        "₀": "_0", "₁": "_1", "₂": "_2", "₃": "_3", "₄": "_4",
        "①": "1)", "②": "2)", "③": "3)", "④": "4)",
        "✓": "check", "✗": "no", "～": "~",
    }))
    text = re.sub(r"\\([_*`\[\]()#+.!-])", r"\1", text)
    text = re.sub(r"[\u2011\u2013\u2014]+", " - ", text)
    parts: list[str] = []
    position = 0
    for match in INLINE_PATTERN.finditer(text):
        parts.append(html.escape(text[position:match.start()]))
        token = match.group(0)
        if token.startswith("**"):
            parts.append(f'<font name="{bold_font}">{html.escape(token[2:-2])}</font>')
        elif token.startswith("`"):
            parts.append(f'<font name="{MONO_FONT}" color="#1F4D78" backColor="#EEF2F6">'
                         f'{html.escape(token[1:-1])}</font>')
        elif token.startswith("*"):
            parts.append(f'<i>{html.escape(token[1:-1])}</i>')
        else:
            link = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", token)
            if link:
                parts.append(f'<a href="{html.escape(link.group(2), quote=True)}" '
                             f'color="#0563C1"><u>{html.escape(link.group(1))}</u></a>')
        position = match.end()
    parts.append(html.escape(text[position:]))
    return "".join(parts)


def _styles(regular: str, bold: str, language: str = "zh") -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    # Chinese text has no inter-word spaces, so JUSTIFY stretches the handful of spaces
    # that exist (around English terms, commas, colons) into ugly rivers. Use LEFT for CJK.
    body_align = TA_LEFT if language == "zh" else TA_JUSTIFY
    return {
        "body": ParagraphStyle(
            "BlogBody", parent=base["BodyText"], fontName=regular, fontSize=10.2,
            leading=15, alignment=body_align, textColor=colors.HexColor("#202124"),
            spaceBefore=0, spaceAfter=7.5, allowWidows=0, allowOrphans=0,
        ),
        "lead": ParagraphStyle(
            "BlogLead", parent=base["BodyText"], fontName=regular, fontSize=11.4,
            leading=16.5, alignment=TA_LEFT, textColor=INK, spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "BlogHeading1", parent=base["Heading1"], fontName=bold, fontSize=16,
            leading=21, textColor=BLUE, spaceBefore=18, spaceAfter=9, keepWithNext=1,
        ),
        "h2": ParagraphStyle(
            "BlogHeading2", parent=base["Heading2"], fontName=bold, fontSize=13,
            leading=18, textColor=BLUE, spaceBefore=13, spaceAfter=6, keepWithNext=1,
        ),
        "h3": ParagraphStyle(
            "BlogHeading3", parent=base["Heading3"], fontName=bold, fontSize=11.5,
            leading=16, textColor=DARK_BLUE, spaceBefore=9, spaceAfter=4, keepWithNext=1,
        ),
        "caption": ParagraphStyle(
            "BlogCaption", parent=base["BodyText"], fontName=regular, fontSize=8.5,
            leading=12, alignment=TA_CENTER, textColor=MUTED, spaceBefore=3, spaceAfter=11,
        ),
        "quote": ParagraphStyle(
            "BlogQuote", parent=base["BodyText"], fontName=regular, fontSize=10.8,
            leading=15.2, alignment=TA_LEFT, textColor=INK, leftIndent=12, rightIndent=8,
            borderColor=BLUE, borderWidth=0, borderLeft=3, borderPadding=(7, 9, 7, 10),
            backColor=LIGHT, spaceBefore=5, spaceAfter=11,
        ),
        "list": ParagraphStyle(
            "BlogList", parent=base["BodyText"], fontName=regular, fontSize=10.2,
            leading=14.5, textColor=colors.HexColor("#202124"), spaceAfter=3.5,
        ),
        "compact_list": ParagraphStyle(
            "BlogCompactList", parent=base["BodyText"], fontName=regular, fontSize=9.4,
            leading=12.8, textColor=colors.HexColor("#202124"), spaceAfter=1.5,
        ),
        "table": ParagraphStyle(
            "BlogTable", parent=base["BodyText"], fontName=regular, fontSize=8.2,
            leading=11, textColor=colors.HexColor("#202124"), alignment=TA_LEFT,
        ),
        "table_header": ParagraphStyle(
            "BlogTableHeader", parent=base["BodyText"], fontName=bold, fontSize=8.3,
            leading=11, textColor=INK, alignment=TA_CENTER,
        ),
        "code_block": ParagraphStyle(
            "BlogCodeBlock", parent=base["Code"], fontName=regular, fontSize=9.2,
            leading=14, textColor=INK, leftIndent=9, rightIndent=9,
            borderColor=BORDER, borderWidth=0.6, borderPadding=8,
            backColor=LIGHT, spaceBefore=5, spaceAfter=10,
        ),
        "formula": ParagraphStyle(
            "BlogFormula", parent=base["BodyText"], fontName=bold, fontSize=10.5,
            leading=15, alignment=TA_CENTER, textColor=INK, borderColor=BORDER,
            borderWidth=0.5, borderPadding=7, backColor=LIGHT,
            spaceBefore=5, spaceAfter=10,
        ),
        "cover_kicker": ParagraphStyle(
            "CoverKicker", parent=base["BodyText"], fontName=bold, fontSize=9.5,
            leading=12, alignment=TA_CENTER, textColor=BLUE, spaceAfter=16,
        ),
        "cover_title": ParagraphStyle(
            "CoverTitle", parent=base["Title"], fontName=bold, fontSize=25,
            leading=34, alignment=TA_CENTER, textColor=INK, spaceAfter=12,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle", parent=base["BodyText"], fontName=regular, fontSize=13,
            leading=19, alignment=TA_CENTER, textColor=DARK_BLUE, spaceAfter=30,
        ),
        "cover_meta": ParagraphStyle(
            "CoverMeta", parent=base["BodyText"], fontName=regular, fontSize=9.5,
            leading=14, alignment=TA_CENTER, textColor=MUTED, spaceAfter=5,
        ),
        "header": ParagraphStyle(
            "PageHeader", parent=base["BodyText"], fontName=regular, fontSize=7.8,
            leading=10, textColor=MUTED,
        ),
    }


class BlogDocTemplate(SimpleDocTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bookmark_index = 0

    def afterFlowable(self, flowable) -> None:  # noqa: N802 - ReportLab hook
        if not isinstance(flowable, Paragraph):
            return
        levels = {"BlogHeading1": 0, "BlogHeading2": 1, "BlogHeading3": 2}
        level = levels.get(flowable.style.name)
        if level is None:
            return
        self._bookmark_index += 1
        key = f"heading-{self._bookmark_index}"
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(flowable.getPlainText(), key, level=level, closed=False)


def _image_flowable(path: Path) -> Image:
    with PILImage.open(path) as source:
        width, height = source.size
    ratio = min((6.5 * inch) / width, (5.65 * inch) / height)
    return Image(str(path), width=width * ratio, height=height * ratio, hAlign="CENTER")


def _table_widths(rows: list[list[str]]) -> list[float]:
    columns = max(len(row) for row in rows)
    scores = []
    for index in range(columns):
        values = [re.sub(r"[*`]", "", row[index]) if index < len(row) else "" for row in rows]
        scores.append(max(4, min(42, max((len(value) for value in values), default=4))))
    minimum = 0.72 * inch if columns > 4 else 0.85 * inch
    available = 6.5 * inch - minimum * columns
    total = sum(scores)
    widths = [minimum + available * score / total for score in scores]
    widths[-1] += 6.5 * inch - sum(widths)
    return widths


def _make_table(rows: list[list[str]], styles: dict[str, ParagraphStyle], regular: str,
                bold: str, language: str = "zh") -> Table:
    columns = max(len(row) for row in rows)
    data = []
    for row_index, row in enumerate(rows):
        style = styles["table_header"] if row_index == 0 else styles["table"]
        data.append([
            Paragraph(
                _cjk_break_html(
                    _inline_markup(row[index] if index < len(row) else "",
                                   regular_font=regular, bold_font=bold))
                if language == "zh" else _inline_markup(row[index] if index < len(row) else "",
                                                         regular_font=regular, bold_font=bold),
                style)
            for index in range(columns)
        ])
    table = Table(data, colWidths=_table_widths(rows), repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
        ("TEXTCOLOR", (0, 0), (-1, 0), INK),
        ("GRID", (0, 0), (-1, -1), 0.55, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    table.setStyle(TableStyle(commands))
    return table


def _collect_list(lines: list[str], start: int, ordered: bool,
                  style: ParagraphStyle, regular: str, bold: str,
                  language: str = "zh") -> tuple[ListFlowable, int]:
    pattern = re.compile(r"^\d+[.)]\s+(.+)$" if ordered else r"^[-+*]\s+(.+)$")
    items = []
    index = start
    while index < len(lines):
        match = pattern.match(lines[index].strip())
        if not match:
            break
        item_lines = [match.group(1).rstrip()]
        index += 1

        # Markdown commonly separates list items with a blank line and uses an
        # indented or plain continuation paragraph after the marker. Keep those
        # lines inside the current ListItem so a 1./2./3. list is not split into
        # three independent one-item lists (which would render as 1./1./1.).
        while index < len(lines):
            stripped = lines[index].strip()
            if pattern.match(stripped):
                break
            if not stripped:
                lookahead = index + 1
                while lookahead < len(lines) and not lines[lookahead].strip():
                    lookahead += 1
                if lookahead < len(lines) and pattern.match(lines[lookahead].strip()):
                    index = lookahead
                    break
                index = lookahead
                break
            if re.match(r"^(?:#{1,6}\s+|[-+*]\s+|\d+[.)]\s+|>|```|---+$)", stripped):
                break
            item_lines.append(stripped)
            index += 1

        markup = _inline_markup(" ".join(item_lines), regular_font=regular, bold_font=bold)
        if language == "zh":
            markup = _cjk_break_html(markup)
        paragraph = Paragraph(markup, style,
        )
        item_value = len(items) + 1 if ordered else None
        items.append(ListItem(paragraph, leftIndent=13, value=item_value))
    flow = ListFlowable(
        items, bulletType="1" if ordered else "bullet", start=1 if ordered else None,
        leftIndent=25,
        bulletFontName=regular, bulletFontSize=9, bulletColor=INK, bulletOffsetY=1,
        spaceBefore=1, spaceAfter=6,
    )
    return flow, index


def _page_header_footer(canvas, document, *, title: str, regular: str, language: str) -> None:
    canvas.saveState()
    canvas.setFont(regular, 7.8)
    canvas.setFillColor(MUTED)
    canvas.drawString(document.leftMargin, LETTER[1] - 0.52 * inch, title[:58])
    footer_label = "Technical Deep Dive" if language == "en" else "技术深度解析"
    canvas.drawRightString(LETTER[0] - document.rightMargin, 0.48 * inch,
                           f"{footer_label}  ·  {canvas.getPageNumber()}")
    canvas.restoreState()


def _first_page(canvas, document) -> None:
    canvas.saveState()
    canvas.setTitle(getattr(document, "blog_title", "技术博客"))
    canvas.setAuthor("Technical Blog Pipeline")
    canvas.setSubject("Technical video and slide synthesis")
    canvas.restoreState()


def markdown_to_pdf(markdown_path: Path, output_path: Path, *, source_label: str | None = None,
                    language: str = "zh") -> Path:
    regular, bold = register_fonts(language)
    styles = _styles(regular, bold, language)
    markdown_path = markdown_path.resolve()
    output_path = output_path.resolve()

    def _styled(text: str) -> str:
        """Render inline markup and, for Chinese, insert CJK word breaks."""
        html = _inline_markup(text, regular_font=regular, bold_font=bold)
        if language == "zh":
            html = _cjk_break_html(html)
        return html

    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    title, subtitle, lines = _extract_title(lines)

    document = BlogDocTemplate(
        str(output_path), pagesize=LETTER, rightMargin=inch, leftMargin=inch,
        topMargin=0.82 * inch, bottomMargin=0.78 * inch,
        title=title, author="Technical Blog Pipeline", subject=subtitle,
    )
    document.blog_title = title
    story: list[Any] = [
        Spacer(1, 1.5 * inch),
        Paragraph("TECHNICAL DEEP DIVE", styles["cover_kicker"]),
        Paragraph(html.escape(title), styles["cover_title"]),
    ]
    if subtitle:
        story.append(Paragraph(html.escape(subtitle), styles["cover_subtitle"]))
    story.extend([
        Spacer(1, 0.45 * inch),
        Paragraph(
            "Synthesized from a technical talk and its accompanying slides"
            if language == "en" else "基于技术演讲与配套材料整理",
            styles["cover_meta"],
        ),
    ])
    if source_label:
        story.append(Paragraph(html.escape(source_label), styles["cover_meta"]))
    story.extend([
        Spacer(1, 0.35 * inch),
        Paragraph(date.today().isoformat(), styles["cover_meta"]),
        PageBreak(),
    ])

    index = 0
    body_count = 0
    current_section = ""
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue
        if stripped == "---":
            story.append(Spacer(1, 3))
            story.append(HRFlowable(width="100%", thickness=0.35, color=BORDER,
                                    spaceBefore=5, spaceAfter=8))
            index += 1
            continue
        if stripped.startswith("```"):
            code_lines: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index].rstrip())
                index += 1
            if index < len(lines):
                index += 1
            code_text = re.sub(r"\\([_*`\[\]()#+.!-])", r"\1", "\n".join(code_lines))
            story.append(Preformatted(code_text, styles["code_block"], maxLineLength=88))
            continue
        if stripped in {r"\[", "$$"}:
            closing = r"\]" if stripped == r"\[" else "$$"
            formula_lines: list[str] = []
            index += 1
            while index < len(lines) and lines[index].strip() != closing:
                formula_lines.append(lines[index].strip())
                index += 1
            if index < len(lines):
                index += 1
            story.append(Paragraph(html.escape(latex_to_plain(" ".join(formula_lines))),
                                   styles["formula"]))
            continue
        if ((stripped.startswith("$$") and stripped.endswith("$$")) or
                (stripped.startswith(r"\[") and stripped.endswith(r"\]"))):
            story.append(Paragraph(html.escape(latex_to_plain(stripped)), styles["formula"]))
            index += 1
            continue
        image_match = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image_match:
            path = (markdown_path.parent / image_match.group(2)).resolve()
            elements: list[Any] = []
            if path.is_file():
                elements.append(_image_flowable(path))
            else:
                elements.append(Paragraph(f"[图片缺失：{html.escape(image_match.group(2))}]",
                                          styles["caption"]))
            if index + 1 < len(lines) and lines[index + 1].strip().startswith("*图"):
                caption = lines[index + 1].strip().strip("*")
                elements.append(Paragraph(_styled(caption), styles["caption"]))
                index += 1
            story.append(KeepTogether(elements))
            index += 1
            continue
        if stripped.startswith("|"):
            rows, index = _parse_table(lines, index)
            story.extend([_make_table(rows, styles, regular, bold, language), Spacer(1, 9)])
            continue
        heading = re.match(r"^(#{2,4})\s+(.+)$", stripped)
        if heading:
            level = min(3, len(heading.group(1)) - 1)
            if level == 1:
                current_section = heading.group(2)
            story.append(Paragraph(_styled(heading.group(2)), styles[f"h{level}"]))
            index += 1
            continue
        if stripped.startswith(">"):
            story.append(Paragraph(_styled(stripped.lstrip("> ")), styles["quote"]))
            index += 1
            continue
        if re.match(r"^[-+*]\s+", stripped):
            list_style = styles["compact_list"] if "结论与局限" in current_section else styles["list"]
            flow, index = _collect_list(lines, index, False, list_style, regular, bold, language)
            story.append(flow)
            continue
        if re.match(r"^\d+[.)]\s+", stripped):
            list_style = styles["compact_list"] if "结论与局限" in current_section else styles["list"]
            flow, index = _collect_list(lines, index, True, list_style, regular, bold, language)
            story.append(flow)
            continue
        if stripped.startswith("*图"):
            story.append(Paragraph(_styled(stripped.strip("*")), styles["caption"]))
            index += 1
            continue
        style = styles["lead"] if body_count == 0 else styles["body"]
        story.append(Paragraph(_styled(stripped), style))
        body_count += 1
        index += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.build(
        story,
        onFirstPage=_first_page,
        onLaterPages=lambda canvas, doc: _page_header_footer(
            canvas, doc, title=title, regular=regular, language=language
        ),
    )
    return output_path


def audit_pdf(path: Path, *, expected_images: int | None = None) -> dict[str, Any]:
    path = path.resolve()
    reader = PdfReader(path)
    page_text = [(page.extract_text() or "").strip() for page in reader.pages]
    image_count = 0
    for page in reader.pages:
        resources = page.get("/Resources") or {}
        xobjects = resources.get("/XObject") or {}
        try:
            values = xobjects.get_object().values()
        except AttributeError:
            values = []
        for value in values:
            try:
                if value.get_object().get("/Subtype") == "/Image":
                    image_count += 1
            except Exception:
                continue
    missing_text_pages = [index + 1 for index, text in enumerate(page_text) if not text]
    passed = (
        path.stat().st_size > 10_000 and len(reader.pages) > 1 and not missing_text_pages
        and (expected_images is None or image_count >= expected_images)
    )
    return {
        "path": str(path), "size": path.stat().st_size, "pass": passed,
        "pages": len(reader.pages), "encrypted": reader.is_encrypted,
        "embedded_image_xobjects": image_count,
        "expected_images": expected_images, "pages_without_extractable_text": missing_text_pages,
        "title": str((reader.metadata or {}).get("/Title", "")),
        "visual_render_qa": "not_run_by_pipeline",
    }


if __name__ == "__main__":
    import argparse
    import json

    cli = argparse.ArgumentParser(description="把图文 Markdown 转为排版好的 PDF")
    cli.add_argument("markdown", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--source-label")
    cli.add_argument("--language", choices=["zh", "en"], default="zh")
    args = cli.parse_args()
    markdown_to_pdf(args.markdown, args.output, source_label=args.source_label,
                    language=args.language)
    print(json.dumps(audit_pdf(args.output), ensure_ascii=False, indent=2))
