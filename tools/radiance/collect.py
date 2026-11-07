#!/usr/bin/env python3
"""Send 8 long source-code documents through Radiance so the LSA dump hook records them."""

import json
import time
import urllib.request
from pathlib import Path

ROOT = Path.home() / "Projects/Radiance"
EXT = {".cpp", ".h", ".hip", ".md", ".py", ".c", ".cmake", ".txt", ".sh"}
# (name, source dirs, target tokens)
DOCS = [
    ("abi+docs", ["abi", "docs"], 32_000),
    ("libquant+tools", ["libquant", "tools"], 48_000),
    ("arch", ["arch"], 64_000),
    ("core", ["core"], 64_000),
    ("libr4d", ["libr4d"], 80_000),
    ("libhalo", ["libhalo"], 96_000),
    ("tests", ["tests"], 96_000),
    ("libref+vendor", ["libref", "vendor"], 96_000),
]


def url(path):
    return "http://127.0.0.1:8000" + path


def count(text):
    r = urllib.request.urlopen(
        urllib.request.Request(
            url("/tokenize"),
            json.dumps({"model": "qwen4exp", "prompt": text}).encode(),
            {"content-type": "application/json"},
        ),
        timeout=120,
    )
    d = json.load(r)
    return d.get("count") or len(d.get("tokens", []))


def build(dirs, target):
    parts, chars = [], 0
    budget = target * 3  # rough chars per token for code, trimmed below
    for d in dirs:
        for f in sorted((ROOT / d).rglob("*")):
            if f.suffix not in EXT or not f.is_file():
                continue
            try:
                t = f.read_text(errors="replace")
            except OSError:
                continue
            parts.append(f"\n===== {f.relative_to(ROOT)} =====\n{t}")
            chars += len(parts[-1])
            if chars > budget * 1.3:
                break
        if chars > budget * 1.3:
            break
    return "".join(parts)


for name, dirs, target in DOCS:
    text = build(dirs, target)
    try:
        n = count(text)
        if n > target:
            text = text[: int(len(text) * target / n)]
    except Exception as e:
        print(f"{name}: tokenize unavailable ({e}), using chars/3", flush=True)
        text = text[: target * 3]
    body = {"model": "qwen4exp", "prompt": text, "max_tokens": 1, "temperature": 0}
    t0 = time.time()
    r = urllib.request.urlopen(
        urllib.request.Request(
            url("/v1/completions"), json.dumps(body).encode(), {"content-type": "application/json"}
        ),
        timeout=7200,
    )
    u = json.load(r)["usage"]
    dt = time.time() - t0
    print(
        f"{name}: {u['prompt_tokens']} tokens in {dt:.0f} s ({u['prompt_tokens'] / dt:.0f} tok/s)", flush=True
    )
print("ALL DONE", flush=True)
