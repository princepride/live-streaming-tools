"""Stage 4: cut the chosen highlight clips with ffmpeg.

Input is either a selection.json (from the Stage 3 review page) or a
*_highlights.json plus --pick "1,3,5". Each chosen clip is cut into:
  - a landscape original-resolution mp4
  - a vertical 9:16 (1080x1920) mp4 with a blurred fill background

Optionally burns sentence subtitles (--burn-subs) built from the Stage 1
transcript segments.

Outputs: <outdir>/<stem>_NN_landscape.mp4 and _vertical.mp4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import log, read_json, stamp  # noqa: E402

VERT_W, VERT_H = 1080, 1920
VBG = (f"[0:v]split=2[bg][fg];"
       f"[bg]scale={VERT_W}:{VERT_H}:force_original_aspect_ratio=increase,"
       f"crop={VERT_W}:{VERT_H},gblur=sigma=20[bg2];"
       f"[fg]scale={VERT_W}:{VERT_H}:force_original_aspect_ratio=decrease[fg2];"
       f"[bg2][fg2]overlay=(W-w)/2:(H-h)/2")


def srt_ts(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


CUE_CHARS = 16          # characters per on-screen subtitle line
BREAK_CHARS = "，。！？；：、,.!?;: "


def split_cue(text: str, limit: int = CUE_CHARS) -> list[str]:
    """Break one ASR segment into short cues.

    Whisper returns segments up to 30s long holding a whole paragraph. Rendered
    as a single subtitle event that becomes a ten-line wall covering the frame,
    so a segment has to be cut into readable cues before it reaches libass.
    """
    parts, current = [], ""
    for ch in text:
        current += ch
        if len(current) >= limit and (ch in BREAK_CHARS or len(current) >= limit + 6):
            parts.append(current.strip(" "))
            current = ""
    if current.strip(" "):
        parts.append(current.strip(" "))
    return [p for p in parts if p] or [text]


def build_srt(segments, start, end, path: Path) -> bool:
    rows, n = [], 0
    for seg in segments:
        s0, s1 = float(seg["start"]), float(seg["end"])
        if s1 <= start or s0 >= end:
            continue
        a = max(s0, start) - start
        b = min(s1, end) - start
        text = (seg.get("text") or "").strip()
        if not text or b <= a:
            continue
        cues = split_cue(text)
        total = sum(len(c) for c in cues) or 1
        cursor = a
        for cue in cues:
            span = (b - a) * len(cue) / total
            n += 1
            rows.append(f"{n}\n{srt_ts(cursor)} --> {srt_ts(cursor + span)}\n{cue}\n")
            cursor += span
    if not rows:
        return False
    path.write_text("\n".join(rows), encoding="utf-8")
    return True


def run_ffmpeg(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{' '.join(cmd)}\n{proc.stderr[-800:]}")


def filter_path(path: Path) -> str:
    """Quote a path for an ffmpeg filtergraph option value.

    A Windows drive letter puts a colon inside the value, which the filtergraph
    parser reads as the separator before the next option -- "C:/x/y.srt" makes
    "/x/y.srt" the value of `original_size`. Quoting alone does not help; the
    colon itself has to be escaped.
    """
    return "'" + path.as_posix().replace("\\", "\\\\").replace(":", r"\:") + "'"


ASS_REF_H = 288  # libass PlayResY for an SRT that declares no script resolution


def probe_height(media: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=height", "-of", "default=nw=1:nk=1", str(media)],
        capture_output=True, text=True)
    try:
        return int(out.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return 1080


def sub_style(video_h: int, font_px: int, margin_px: int) -> str:
    """Build force_style in script units for a target size in real pixels.

    libass scales an SRT's style values by video_height / PlayResY, and a bare
    SRT has no PlayRes, so a literal `FontSize=42` renders at 42 * (1920/288)
    = 280px on a vertical clip. Everything therefore has to be expressed back
    in the 288-tall reference space.
    """
    scale = ASS_REF_H / max(1, video_h)
    return (f"FontSize={max(6, round(font_px * scale))},"
            f"MarginV={max(0, round(margin_px * scale))},"
            f"Outline=1,Shadow=0,Alignment=2")


def cut_landscape(media, start, dur, out, subs_path):
    vf = None
    if subs_path:
        # Sit above whatever subtitles the source frame already carries.
        style = sub_style(probe_height(media), font_px=44, margin_px=130)
        vf = (f"subtitles={filter_path(subs_path)}"
              f":force_style='{style}'")
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(media),
           "-t", f"{dur:.3f}"]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)]
    run_ffmpeg(cmd)


def cut_vertical(media, start, dur, out, subs_path):
    chain = VBG
    if subs_path:
        # The 16:9 frame is centred in the 1080x1920 canvas, so the blurred band
        # below it is free space: park the text there rather than on top of the
        # subtitles the source frame already carries.
        style = sub_style(VERT_H, font_px=58, margin_px=380)
        chain += (f",subtitles={filter_path(subs_path)}"
                  f":force_style='{style}'")
    chain += "[v]"
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(media),
           "-t", f"{dur:.3f}", "-filter_complex", chain, "-map", "[v]", "-map", "0:a?",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)]
    run_ffmpeg(cmd)


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 4: cut selected highlight clips")
    p.add_argument("input", help="selection.json OR *_highlights.json")
    p.add_argument("--pick", help="comma ids to cut when input is *_highlights.json")
    p.add_argument("--media", help="override source media path")
    p.add_argument("--outdir", default=None, help="output dir (default <media_dir>/clips)")
    p.add_argument("--no-landscape", action="store_true")
    p.add_argument("--no-vertical", action="store_true")
    p.add_argument("--burn-subs", action="store_true",
                   help="burn sentence subtitles from the transcript")
    p.add_argument("--transcript", help="*_highlights_transcript.json for --burn-subs")
    args = p.parse_args()

    data = read_json(Path(args.input))
    clips = data.get("clips", [])
    if args.pick:
        want = {int(x) for x in args.pick.replace(" ", "").split(",") if x}
        clips = [c for c in clips if int(c.get("id", -1)) in want]
    elif "picks" in data and data.get("picks"):
        want = set(data["picks"])
        clips = [c for c in clips if int(c.get("id", -1)) in want]
    if not clips:
        p.error("no clips selected (use --pick or a selection.json with picks)")

    media = Path(args.media or data.get("media") or "")
    if not media.exists():
        p.error(f"media not found: {media} (pass --media)")

    segments = None
    if args.burn_subs:
        tpath = args.transcript or data.get("source_transcript")
        if tpath and Path(tpath).exists():
            segments = read_json(Path(tpath)).get("segments")
        else:
            log("warning: --burn-subs set but transcript not found; skipping subs")

    outdir = Path(args.outdir) if args.outdir else media.parent / "clips"
    outdir.mkdir(parents=True, exist_ok=True)
    stem = media.stem

    made = []
    with tempfile.TemporaryDirectory() as tmp:
        for c in clips:
            cid = int(c["id"]); start = float(c["start"]); dur = float(c["duration"])
            subs = None
            if segments:
                sp = Path(tmp) / f"clip_{cid:02d}.srt"
                subs = sp if build_srt(segments, start, c["end"], sp) else None
            log(f"clip #{cid} {stamp(start)}–{stamp(c['end'])} ({dur:.0f}s)")
            if not args.no_landscape:
                out = outdir / f"{stem}_{cid:02d}_landscape.mp4"
                cut_landscape(media, start, dur, out, subs)
                made.append(out)
            if not args.no_vertical:
                out = outdir / f"{stem}_{cid:02d}_vertical.mp4"
                cut_vertical(media, start, dur, out, subs)
                made.append(out)

    log(f"done: {len(made)} files in {outdir}")
    for m in made:
        print(m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
