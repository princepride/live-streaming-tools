#!/usr/bin/env python3
"""Shared speech-to-text core for the stream-tools repo.

One OpenRouter-first transcription implementation used by both:
  - auto_chapters.py           mode="chunk"  (one text block per chunk; the
                               coarse chunk-boundary timing chapters rely on)
  - make-highlight-clips skill mode="fine"   (real segment/word timestamps,
                               accurate enough to cut ~60s clips)

OpenRouter's /audio/transcriptions returns segment timestamps with
response_format=verbose_json when routed to an OpenAI-compatible provider; the
default model openai/whisper-large-v3 is hosted by Groq/Together and qualifies,
so both callers can share one OPENROUTER_API_KEY.
"""
from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# backend -> (default base_url, default model, api key env var)
BACKENDS = {
    "openrouter": ("https://openrouter.ai/api/v1", "openai/whisper-large-v3",
                   "OPENROUTER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "whisper-large-v3", "GROQ_API_KEY"),
    "openai": ("https://api.openai.com/v1", "whisper-1", "OPENAI_API_KEY"),
    "local": (None, "large-v3", None),
}
DEFAULT_BACKEND = "openrouter"


def log(msg: str) -> None:
    print(f"[stt] {msg}", file=sys.stderr, flush=True)


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{proc.stderr}")
    return proc.stdout


def media_duration(path: Path) -> float:
    out = _run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    return float(out.strip())


def extract_audio(media: Path, out: Path, start: float, length: float) -> None:
    _run([
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(media),
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "libmp3lame", "-q:a", "4", str(out),
    ])


def resolve(backend: str, base_url: str | None, model: str | None):
    """Return (base_url, model, api_key) for a backend, reading env/args."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend '{backend}'")
    dbase, dmodel, key_env = BACKENDS[backend]
    base_url = base_url or dbase
    model = model or dmodel
    api_key = None
    if key_env:
        api_key = os.environ.get(key_env)
        if not api_key and backend != "local":
            raise RuntimeError(f"{key_env} not set (needed for backend '{backend}')")
    return base_url, model, api_key


def _post_transcription(base_url: str, api_key: str, model: str, audio: Path,
                        language: str, verbose: bool, want_words: bool,
                        retries: int = 4) -> dict:
    """POST one audio file to an OpenAI-compatible /audio/transcriptions."""
    import requests

    error = "unknown error"
    for attempt in range(retries):
        try:
            with open(audio, "rb") as fh:
                data = {"model": model}
                if language:
                    data["language"] = language
                if verbose:
                    data["response_format"] = "verbose_json"
                    grans = ["segment"] + (["word"] if want_words else [])
                    data["timestamp_granularities[]"] = grans
                resp = requests.post(
                    f"{base_url}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    files={"file": (audio.name, fh, "audio/mpeg")},
                    data=data, timeout=600,
                )
            if resp.status_code < 400:
                return resp.json()
            detail = resp.text[:400]
            if resp.status_code not in {408, 409, 429, 500, 502, 503, 504}:
                raise RuntimeError(f"ASR HTTP {resp.status_code}: {detail}")
            error = f"ASR HTTP {resp.status_code}: {detail}"
        except (requests.RequestException, ValueError) as exc:
            error = str(exc)
        if attempt + 1 < retries:
            delay = 2 ** attempt
            log(f"transcription retry in {delay}s: {error}")
            time.sleep(delay)
    raise RuntimeError(error)


def _chunk_segments(payload: dict, offset: float, mode: str, chunk_end: float):
    """Turn one chunk's API payload into a list of absolute-timed segments."""
    if mode == "fine":
        segs = payload.get("segments") or []
        out = []
        for s in segs:
            item = {"start": round(float(s["start"]) + offset, 3),
                    "end": round(float(s["end"]) + offset, 3),
                    "text": (s.get("text") or "").strip()}
            if s.get("words"):
                item["words"] = [{"start": round(float(w["start"]) + offset, 3),
                                  "end": round(float(w["end"]) + offset, 3),
                                  "word": w.get("word", "")} for w in s["words"]]
            if item["text"]:
                out.append(item)
        if out:
            return out
        # fall through to whole-chunk text if provider returned no segments
    text = (payload.get("text") or "").strip()
    return [{"start": round(offset, 3), "end": round(chunk_end, 3), "text": text}] \
        if text else []


def _transcribe_local(media: Path, model: str, language: str, mode: str,
                      chunk_seconds: int, want_words: bool) -> list[dict]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise SystemExit("faster-whisper not installed. Run: pip install faster-whisper")
    log(f"loading local faster-whisper '{model}' (first run downloads it)")
    wm = WhisperModel(model, device="auto", compute_type="auto")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        _run(["ffmpeg", "-y", "-v", "error", "-i", str(media),
              "-vn", "-ac", "1", "-ar", "16000", str(wav)])
        seg_iter, _ = wm.transcribe(str(wav), language=language or None,
                                    word_timestamps=(mode == "fine" and want_words),
                                    vad_filter=True)
        fine = []
        for s in seg_iter:
            item = {"start": round(s.start, 3), "end": round(s.end, 3),
                    "text": s.text.strip()}
            if mode == "fine" and want_words and s.words:
                item["words"] = [{"start": round(w.start, 3), "end": round(w.end, 3),
                                  "word": w.word} for w in s.words]
            fine.append(item)
    if mode == "fine":
        return [s for s in fine if s["text"]]
    # chunk mode: bin fine segments into chunk_seconds windows
    bins: dict[int, list[str]] = {}
    for s in fine:
        b = int(s["start"] // chunk_seconds)
        bins.setdefault(b, []).append(s["text"])
    total = max(bins) + 1 if bins else 0
    return [{"start": round(b * chunk_seconds, 3),
             "end": round((b + 1) * chunk_seconds, 3),
             "text": " ".join(bins.get(b, []))} for b in range(total)]


def transcribe(media, *, backend: str = DEFAULT_BACKEND, base_url: str | None = None,
               model: str | None = None, language: str = "zh",
               chunk_seconds: int = 240, mode: str = "chunk",
               want_words: bool = False, cache_dir=None, workers: int = 3) -> list[dict]:
    """Transcribe media into a list of segments with absolute timestamps.

    mode="chunk": one {start,end,text} per chunk_seconds window (chapter timing).
    mode="fine":  real segment timestamps (+words if want_words) for clipping.
    Segments are cached per chunk in cache_dir when provided (API backends only).
    """
    media = Path(media)
    if mode not in {"chunk", "fine"}:
        raise ValueError("mode must be 'chunk' or 'fine'")
    base_url, model, api_key = resolve(backend, base_url, model)
    duration = media_duration(media)

    if backend == "local":
        return _transcribe_local(media, model, language, mode, chunk_seconds, want_words)

    n = int((duration + chunk_seconds - 1) // chunk_seconds)
    spans = [(i, i * chunk_seconds, min(duration, (i + 1) * chunk_seconds))
             for i in range(n)]
    log(f"duration {duration:.1f}s, {len(spans)} chunks, backend={backend}, "
        f"model={model}, mode={mode}")
    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    def do_chunk(i: int, start: float, end: float) -> list[dict]:
        import json as _json
        cache = cache_dir / f"part_{i:04d}.json" if cache_dir else None
        if cache and cache.exists():
            return _json.loads(cache.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / f"part_{i:04d}.mp3"
            extract_audio(media, clip, start, end - start)
            payload = _post_transcription(base_url, api_key, model, clip, language,
                                          verbose=(mode == "fine"),
                                          want_words=want_words)
        segs = _chunk_segments(payload, start, mode, end)
        if cache:
            cache.write_text(_json.dumps(segs, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        return segs

    results: dict[int, list[dict]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(do_chunk, i, s, e): i for i, s, e in spans}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            results[futs[fut]] = fut.result()
            done += 1
            log(f"transcribed {done}/{len(spans)}")

    transcript: list[dict] = []
    for i in range(len(spans)):
        transcript.extend(results.get(i, []))
    return transcript
