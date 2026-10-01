#!/usr/bin/env python3
"""把一篇博客导出成适合「复制 → 粘贴到知乎编辑器」的单页 HTML。

知乎的文档导入会丢图、丢排版，直接从网页复制效果更好；但 MkDocs 页面里的
MathJax 公式复制过去会变成乱码。这个脚本生成一个干净的页面：

- 公式改写成知乎自己的公式图片（https://www.zhihu.com/equation?tex=...，
  带 eeimg="1"），这与知乎文章内部的公式标记一致；含中文的公式或加了
  --plain-math 时，改写成普通 Unicode 文本。
- 图片默认指向已发布的 GitHub Pages 地址，由知乎粘贴时自动转存；
  加 --embed-images 则内联为 base64，适合知乎抓不到外链图片的情况。
- 标题单独放在页面顶部（知乎标题是独立输入框），正文不重复。

用法：
    python tools/zhihu_html.py kimi_k3_production_serving
    python tools/zhihu_html.py kimi_k3_production_serving --embed-images
    python tools/zhihu_html.py --all

输出：tech_blog_output/<slug>/zhihu.html，随 MkDocs 站点发布到
https://princepride.github.io/live-streaming-tools/<slug>/zhihu.html。
知乎只会稳定转存从线上网页复制来的图片，所以请从发布后的地址打开，
点「复制标题」「复制正文」，到知乎编辑器里粘贴。
"""

from __future__ import annotations

import argparse
import base64
import html
import mimetypes
import re
import sys
from pathlib import Path
from urllib.parse import quote

import markdown

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from blog_docx import latex_to_plain  # noqa: E402

BLOGS = ROOT / "tech_blog_output"
SITE_URL = "https://princepride.github.io/live-streaming-tools/"

FENCE = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.M | re.S)
INLINE_CODE = re.compile(r"`[^`\n]+`")
MATH = re.compile(
    r"\\\[(?P<db>.+?)\\\]"                      # \[ ... \]
    r"|\$\$(?P<dd>.+?)\$\$"                     # $$ ... $$
    r"|\\\((?P<ib>.+?)\\\)"                     # \( ... \)
    r"|(?<![\\$])\$(?P<id>[^\s$](?:[^$\n]*[^\s$])?)\$(?!\d)",  # $ ... $
    re.S,
)
CJK = re.compile(r"[　-〿一-鿿＀-￯]")


def _equation_img(tex: str, display: bool) -> str:
    tex = " ".join(tex.split())
    # Zhihu renders \text{...} literally, so an escaped underscore would show its backslash.
    tex = re.sub(r"\\text\{([^{}]*)\}", lambda m: r"\text{" + m.group(1).replace(r"\_", "_") + "}", tex)
    # A trailing "\\" is how Zhihu's own markup marks a display (block) formula.
    src = "https://www.zhihu.com/equation?tex=" + quote(tex + (r"\\" if display else ""), safe="")
    return (f'<img src="{src}" alt="{html.escape(tex, quote=True)}" '
            f'class="ee_img tr_noresize" eeimg="1">')


def _math_html(tex: str, display: bool, plain: bool) -> str:
    if plain or CJK.search(tex):
        text = html.escape(latex_to_plain(tex))
        return f'<p class="formula">{text}</p>' if display else text
    img = _equation_img(tex, display)
    return f'<p class="formula">{img}</p>' if display else img


def _protect_math(source: str, plain: bool) -> tuple[str, dict[str, str]]:
    """Swap math for placeholders before Markdown sees it, skipping code."""
    stash: dict[str, str] = {}

    def math(match: re.Match[str]) -> str:
        display = match.group("db") is not None or match.group("dd") is not None
        tex = next(g for g in match.group("db", "dd", "ib", "id") if g is not None)
        token = f"ZHMATH{len(stash):04d}ZH"
        stash[token] = _math_html(tex, display, plain)
        return f"\n\n{token}\n\n" if display else token

    # Code must reach Markdown untouched, so park it first and restore it after.
    code: dict[str, str] = {}

    def park(match: re.Match[str]) -> str:
        token = f"ZHCODE{len(code):04d}ZH"
        code[token] = match.group(0)
        return token

    text = FENCE.sub(park, source)
    text = INLINE_CODE.sub(park, text)
    text = MATH.sub(math, text)
    for token, fragment in code.items():
        text = text.replace(token, fragment)
    return text, stash


def _split_title(source: str) -> tuple[str, str]:
    lines = source.splitlines()
    title = ""
    if lines and lines[0].startswith("# "):
        title = lines.pop(0)[2:].strip()
    return title, "\n".join(lines)


def _image_src(src: str, final_dir: Path, slug: str, embed: bool) -> str:
    if re.match(r"^(https?:|data:)", src):
        return src
    path = (final_dir / src).resolve()
    if embed and path.is_file():
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
    return f"{SITE_URL}{slug}/final/{src}"


def render(slug: str, *, english: bool = False, embed: bool = False, plain_math: bool = False) -> Path:
    final_dir = BLOGS / slug / "final"
    source_path = final_dir / ("blog.en.md" if english else "blog.md")
    title, body = _split_title(source_path.read_text(encoding="utf-8"))
    body, stash = _protect_math(body, plain_math)
    article = markdown.markdown(
        body, extensions=["tables", "fenced_code", "footnotes", "sane_lists"], output_format="html")
    for token, fragment in stash.items():
        article = article.replace(f"<p>{token}</p>", fragment).replace(token, fragment)
    article = re.sub(
        r'(<img\b[^>]*?\bsrc=")([^"]+)(")',
        lambda m: m.group(1) + _image_src(m.group(2), final_dir, slug, embed) + m.group(3),
        article,
    )
    # Keep images as plain <p><img></p>, exactly like the MkDocs page: Zhihu's paste
    # handler treats <figure> as its own internal markup and drops pasted ones.
    article = re.sub(r"<p>(<img\b(?![^>]*eeimg)[^>]*>)</p>", r'<p class="pic">\1</p>', article)

    page = PAGE.format(
        lang="en" if english else "zh-CN",
        title=html.escape(title),
        title_attr=html.escape(title, quote=True),
        article=article,
        mode=("图片已内联" if embed else "图片使用 GitHub Pages 外链")
        + ("，公式为纯文本" if plain_math else "，公式为知乎公式图片"),
    )
    out = final_dir.parent / ("zhihu.en.html" if english else "zhihu.html")
    out.write_text(page, encoding="utf-8")
    return out


PAGE = """<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ --ink:#1a1a1a; --muted:#8590a6; --line:#ebebeb; --accent:#056de8; --code:#f6f6f6; --bg:#fff; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
         font:16px/1.75 -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif; }}
  .bar {{ position:sticky; top:0; z-index:1; display:flex; flex-wrap:wrap; gap:8px; align-items:center;
          padding:10px 16px; background:#f8f9fb; border-bottom:1px solid var(--line); font-size:14px; }}
  .bar button {{ padding:6px 14px; border:1px solid var(--accent); border-radius:4px; cursor:pointer;
                 background:var(--accent); color:#fff; font-size:14px; }}
  .bar button.ghost {{ background:#fff; color:var(--accent); }}
  .bar .hint {{ color:var(--muted); }}
  main {{ max-width:690px; margin:0 auto; padding:24px 16px 80px; }}
  h1.title {{ font-size:26px; line-height:1.4; margin:8px 0 20px; }}
  #article h2 {{ font-size:22px; margin:32px 0 12px; }}
  #article h3 {{ font-size:19px; margin:24px 0 10px; }}
  #article img {{ max-width:100%; }}
  #article img[eeimg] {{ max-width:none; vertical-align:middle; }}
  #article p.pic {{ margin:16px 0 4px; text-align:center; }}
  #article .formula {{ text-align:center; overflow-x:auto; }}
  #article blockquote {{ margin:16px 0; padding:0 14px; border-left:3px solid #d3d3d3; color:#646464; }}
  #article pre {{ background:var(--code); padding:12px 14px; overflow-x:auto; border-radius:4px; font-size:14px; }}
  #article code {{ background:var(--code); padding:1px 4px; border-radius:3px; font-size:0.92em; }}
  #article pre code {{ background:none; padding:0; }}
  #article table {{ border-collapse:collapse; margin:16px 0; font-size:14px; display:block; overflow-x:auto; }}
  #article th, #article td {{ border:1px solid var(--line); padding:6px 10px; vertical-align:top; }}
  #article th {{ background:#f6f6f6; }}
  #article hr {{ border:0; border-top:1px solid var(--line); margin:28px 0; }}
</style>
</head>
<body>
<div class="bar">
  <button id="copy-body">复制正文</button>
  <button id="copy-title" class="ghost">复制标题</button>
  <span class="hint" id="status">{mode}。在知乎编辑器正文处粘贴（Ctrl/⌘+V）。</span>
</div>
<main>
  <h1 class="title" id="title">{title}</h1>
  <article id="article">
{article}
  </article>
</main>
<script>
  const status = document.getElementById("status");
  // Use the browser's native selection copy, the same path as copying from the
  // website by hand; Zhihu only re-hosts pasted images reliably from that payload.
  function copyNode(node) {{
    const range = document.createRange();
    range.selectNodeContents(node);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    let ok = false;
    try {{ ok = document.execCommand("copy"); }} catch (err) {{ ok = false; }}
    status.textContent = ok ? "已复制，去知乎编辑器粘贴即可。" : "已选中内容，请按 Ctrl/⌘+C 复制。";
    if (ok) selection.removeAllRanges();
  }}
  document.getElementById("copy-body").onclick = () => copyNode(document.getElementById("article"));
  document.getElementById("copy-title").onclick = () => copyNode(document.getElementById("title"));
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="导出适合粘贴到知乎的博客页面")
    parser.add_argument("slugs", nargs="*", help="tech_blog_output 下的博客目录名")
    parser.add_argument("--all", action="store_true", help="导出所有含 final/blog.md 的博客")
    parser.add_argument("--en", action="store_true", help="同时导出英文版")
    parser.add_argument("--embed-images", action="store_true", help="图片内联为 base64")
    parser.add_argument("--plain-math", action="store_true", help="公式全部改写为 Unicode 文本")
    args = parser.parse_args()
    slugs = sorted(p.parent.parent.name for p in BLOGS.glob("*/final/blog.md")) if args.all else args.slugs
    if not slugs:
        parser.error("请给出博客目录名，或使用 --all")
    for slug in slugs:
        for english in ([False, True] if args.en else [False]):
            out = render(slug, english=english, embed=args.embed_images, plain_math=args.plain_math)
            print(out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
