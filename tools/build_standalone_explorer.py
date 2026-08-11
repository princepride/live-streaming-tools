#!/usr/bin/env python3
"""把 Parallelism Explorer 打包成一个可离线双击打开的单文件 HTML。

文档站上的页面（tech_blog_output/parallelism_explorer/index.md）与这个离线版
共用同一份 css / js，这里只是把它们内联进一个自带明暗切换的壳子里，
避免两份实现漂移。

用法：
    python tools/build_standalone_explorer.py
    python tools/build_standalone_explorer.py -o /path/to/out.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "tech_blog_output" / "assets"
CSS = ASSETS / "parallelism-explorer.css"
JS = ASSETS / "parallelism-explorer.js"
DEFAULT_OUT = ROOT / "vllm-parallel-layer-viz.html"

SHELL = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Parallelism Explorer · 并行切分可视化</title>
<style>
/* 离线版自带一份最小主题变量，对齐文档站 extra.css 的取值 */
:root {{
  --md-text-font: "Inter", "Noto Sans SC", system-ui, sans-serif;
  --md-code-font: "JetBrains Mono", "Cascadia Code", ui-monospace, monospace;
  --vllm-ink: #111827; --vllm-slate: #334155; --vllm-muted: #64748b;
  --vllm-line: #e2e8f0; --vllm-paper: #f8fafc; --vllm-accent: #3b82f6;
}}
body[data-md-color-scheme="slate"] {{
  --vllm-ink: #f8fafc; --vllm-slate: #dbe4ef; --vllm-muted: #94a3b8;
  --vllm-line: #263244; --vllm-paper: #121923; --vllm-accent: #60a5fa;
}}
html, body {{ margin: 0; padding: 0; }}
body {{
  background: var(--vllm-paper);
  color: var(--vllm-ink);
  font-family: var(--md-text-font);
  font-size: 16px;
}}
body[data-md-color-scheme="slate"] {{ background: #0b1119; }}
.page {{ max-width: 84rem; margin: 0 auto; padding: 1.6rem 1.2rem 5rem; }}
.themebtn {{
  position: fixed; top: .8rem; right: .9rem; z-index: 9;
  padding: .32rem .8rem; border: 1px solid var(--vllm-line); border-radius: .4rem;
  background: var(--vllm-paper); color: var(--vllm-muted);
  font-family: var(--md-code-font); font-size: .72rem; font-weight: 700; cursor: pointer;
}}
</style>
<style data-pe-style="1">
{css}
</style>
</head>
<body data-md-color-scheme="default">
<button class="themebtn" id="themebtn">◐ theme</button>
<div class="page"><div id="pe-root"></div></div>
<script>
(function () {{
  var b = document.body, k = "pe-scheme";
  try {{ var s = localStorage.getItem(k); if (s) b.setAttribute("data-md-color-scheme", s); }} catch (e) {{}}
  document.getElementById("themebtn").addEventListener("click", function () {{
    var next = b.getAttribute("data-md-color-scheme") === "slate" ? "default" : "slate";
    b.setAttribute("data-md-color-scheme", next);
    try {{ localStorage.setItem(k, next); }} catch (e) {{}}
  }});
}})();
</script>
<script>
{js}
</script>
</body>
</html>
"""


def build(out: Path) -> Path:
    css = CSS.read_text(encoding="utf-8")
    js = JS.read_text(encoding="utf-8")
    out.write_text(SHELL.format(css=css, js=js), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT, help="输出的单文件 HTML 路径")
    args = ap.parse_args()
    for src in (CSS, JS):
        if not src.exists():
            raise SystemExit(f"missing source asset: {src}")
    path = build(args.out)
    print(f"wrote {path}  ({path.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
