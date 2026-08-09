"""Shared helpers for the make-highlight-clips skill."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def log(msg: str) -> None:
    print(f"[highlight-clips] {msg}", file=sys.stderr, flush=True)


def run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{proc.stderr}")
    return proc.stdout


def media_duration(path: Path) -> float:
    out = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    return float(out.strip())


def stamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def load_env_file(root: Path) -> None:
    """Load KEY=VALUE pairs from a git-ignored .env without overriding real env."""
    env_path = root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


import re as _re
import time as _time


def request_json(url: str, headers: dict, payload: dict, timeout: int = 300,
                 retries: int = 4) -> dict:
    """POST JSON with retry/backoff on transient errors (OpenAI-compatible)."""
    import requests
    error = "unknown error"
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code < 400:
                return resp.json()
            detail = resp.text[:500]
            if resp.status_code not in {408, 409, 429, 500, 502, 503, 504}:
                raise RuntimeError(f"HTTP {resp.status_code}: {detail}")
            error = f"HTTP {resp.status_code}: {detail}"
        except (requests.RequestException, ValueError) as exc:
            error = str(exc)
        if attempt + 1 < retries:
            delay = 2 ** attempt
            log(f"request failed, retry in {delay}s: {error}")
            _time.sleep(delay)
    raise RuntimeError(error)


def parse_json_content(content: str):
    """Extract a JSON object/array from a model reply that may be fenced."""
    content = (content or "").strip()
    if content.startswith("```"):
        content = _re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=_re.I | _re.S)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = _re.search(r"[\{\[].*[\}\]]", content, _re.S)
        if not match:
            raise RuntimeError("model did not return parseable JSON")
        return json.loads(match.group(0))


def chat_json(base_url: str, api_key: str, model: str, prompt: str,
              title: str = "make-highlight-clips", attempts: int = 3) -> dict:
    """Call an OpenAI-compatible chat endpoint, return parsed JSON content.

    request_json already retries transport-level failures. This loop covers the
    other way a call fails: HTTP 200 carrying an empty or non-JSON message, which
    a long reasoning model emits often enough to matter over a multi-window run.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": title,
    }
    error = "unknown error"
    for attempt in range(attempts):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2 + 0.1 * attempt,
            "response_format": {"type": "json_object"},
        }
        result = request_json(f"{base_url}/chat/completions", headers, payload)
        choice = (result.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        try:
            return parse_json_content(content)
        except (RuntimeError, ValueError) as exc:
            reason = choice.get("finish_reason", "?")
            error = f"{exc} (finish_reason={reason}, content={content[:200]!r})"
            if attempt + 1 < attempts:
                log(f"unparseable reply, retrying: {error}")
    raise RuntimeError(error)
