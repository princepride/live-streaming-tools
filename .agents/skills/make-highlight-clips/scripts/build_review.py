"""Stage 3: build a self-contained HTML review page for highlight candidates.

Reads <stem>_highlights.json and renders checkbox cards. The user ticks the clips
they want, then clicks "导出所选" to download <stem>_selection.json (and can copy a
--pick string for the CLI). The page needs no server and no network.

Output: <stem>_highlights_review.html
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import log, read_json  # noqa: E402

PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>金句片段挑选 — __TITLE__</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
         max-width: 860px; margin: 0 auto; padding: 24px; line-height: 1.6; }
  header { position: sticky; top: 0; background: Canvas; padding: 12px 0;
           border-bottom: 1px solid #8884; z-index: 10; }
  h1 { font-size: 20px; margin: 0 0 8px; }
  .bar { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  button { font-size: 14px; padding: 7px 14px; border-radius: 8px;
           border: 1px solid #8886; background: #6663; cursor: pointer; }
  button.primary { background: #2563eb; color: #fff; border-color: #2563eb; }
  .count { margin-left: auto; opacity: .75; font-size: 14px; }
  .card { border: 1px solid #8884; border-radius: 12px; padding: 14px 16px;
          margin: 14px 0; }
  .card.sel { border-color: #2563eb; background: #2563eb14; }
  .top { display: flex; gap: 12px; align-items: flex-start; }
  .top input { margin-top: 6px; width: 18px; height: 18px; }
  .hook { font-size: 17px; font-weight: 650; }
  .meta { font-size: 13px; opacity: .8; margin: 4px 0 2px; }
  .score { font-weight: 700; color: #2563eb; }
  .reason { font-size: 14px; }
  details { margin-top: 8px; font-size: 14px; }
  summary { cursor: pointer; opacity: .8; }
  blockquote { margin: 8px 0 0; padding-left: 12px; border-left: 3px solid #8886;
               white-space: pre-wrap; }
  .pick { font-family: ui-monospace, monospace; background: #6663; padding: 2px 6px;
          border-radius: 6px; }
</style>
</head>
<body>
<header>
  <h1>金句片段挑选 — __TITLE__</h1>
  <div class="bar">
    <button onclick="all(true)">全选</button>
    <button onclick="all(false)">清空</button>
    <button class="primary" onclick="exportSel()">导出所选 (selection.json)</button>
    <button onclick="copyPick()">复制 --pick</button>
    <span class="count" id="count"></span>
  </div>
  <div class="meta">剪辑命令：<span class="pick" id="pickstr">--pick</span></div>
</header>
<main id="cards"></main>
<script>
const CLIPS = __DATA__;
const META = __META__;
const sel = new Set();

function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML;}
function render(){
  const box = document.getElementById('cards');
  box.innerHTML = '';
  CLIPS.forEach(c => {
    const card = document.createElement('div');
    card.className = 'card' + (sel.has(c.id) ? ' sel' : '');
    card.innerHTML =
      '<div class="top"><input type="checkbox" ' + (sel.has(c.id)?'checked':'') +
      ' data-id="'+c.id+'">' +
      '<div><div class="hook">#'+c.id+'  「'+esc(c.hook)+'」</div>' +
      '<div class="meta">'+c.start_ts+' – '+c.end_ts+' · '+Math.round(c.duration)+
      ' 秒 · <span class="score">'+c.score+'/100</span></div>' +
      '<div class="reason">'+esc(c.reason)+'</div>' +
      '<details><summary>完整对话</summary><blockquote>'+esc(c.transcript)+
      '</blockquote></details></div></div>';
    box.appendChild(card);
  });
  box.querySelectorAll('input[type=checkbox]').forEach(cb => {
    cb.onchange = () => { const id=+cb.dataset.id;
      cb.checked ? sel.add(id) : sel.delete(id); render(); };
  });
  const picks = [...sel].sort((a,b)=>a-b);
  document.getElementById('count').textContent = '已选 '+picks.length+' / '+CLIPS.length;
  document.getElementById('pickstr').textContent = '--pick ' + (picks.join(',') || '');
}
function all(on){ sel.clear(); if(on) CLIPS.forEach(c=>sel.add(c.id)); render(); }
function copyPick(){
  const picks=[...sel].sort((a,b)=>a-b).join(',');
  navigator.clipboard.writeText('--pick '+picks); }
function exportSel(){
  const picks=[...sel].sort((a,b)=>a-b);
  const chosen=CLIPS.filter(c=>sel.has(c.id));
  const payload={media:META.media, source_highlights:META.source_highlights,
                 picks:picks, clips:chosen};
  const blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download=META.selection_name;
  a.click();
}
render();
</script>
</body>
</html>
"""


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 3: build HTML review page")
    p.add_argument("highlights", help="*_highlights.json from Stage 2")
    p.add_argument("-o", "--output", help="output HTML path")
    args = p.parse_args()

    hpath = Path(args.highlights)
    if not hpath.exists():
        p.error(f"highlights not found: {hpath}")
    data = read_json(hpath)
    clips = data.get("clips", [])
    media = data.get("media") or ""
    title = html.escape(Path(media).name or hpath.stem)

    selection_name = hpath.name.replace("_highlights.json", "_selection.json")
    if selection_name == hpath.name:
        selection_name = hpath.stem + "_selection.json"
    meta = {"media": media, "source_highlights": str(hpath),
            "selection_name": selection_name}

    page = (PAGE.replace("__TITLE__", title)
                .replace("__DATA__", json.dumps(clips, ensure_ascii=False))
                .replace("__META__", json.dumps(meta, ensure_ascii=False)))

    out_path = Path(args.output) if args.output else \
        hpath.with_name(hpath.stem + "_review.html")
    out_path.write_text(page, encoding="utf-8")
    log(f"wrote review page ({len(clips)} clips) -> {out_path}")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
