#!/usr/bin/env python3
"""Needle in a haystack against a running Radiance server: a passcode hidden at a depth of a filler prompt.

Each case is `repeats:depth` - the filler sentence repeated `repeats` times (about 18 tokens each), the needle
inserted at `depth` of it. Prints the prompt length, whether the answer holds the passcode, and the server's timings.

    needle.py 500:0.5 2000:0.1 2000:0.5 2000:0.9 8000:0.25 8000:0.75
"""

import argparse
import json
import random
import time
import urllib.request

FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "


def ask(url: str, repeats: int, depth: float) -> None:
    """Ask for the passcode hidden at `depth` of `repeats` filler sentences and print the result."""
    parts = [FILLER] * repeats
    code = random.randint(10000, 99999)
    parts.insert(int(len(parts) * depth), f"The secret passcode for the blue door is {code}. Remember it. ")
    prompt = (
        "".join(parts) + "\n\nWhat is the secret passcode for the blue door? Answer with the number only."
    )
    body = {
        "model": "qwen4exp",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 600,
        "temperature": 0,
    }
    request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    start = time.time()
    reply = json.load(urllib.request.urlopen(request, timeout=3600))
    answer = (reply["choices"][0]["message"].get("content") or "").strip()
    timings = reply.get("timings", {})
    print(
        f"ctx {reply['usage']['prompt_tokens']:7d} depth {depth:.2f} want {code} got {answer[:40]!r} "
        f"{'OK' if str(code) in answer else 'MISS'}  prefill {timings.get('prompt_per_second', 0):.0f} tok/s  "
        f"decode {timings.get('predicted_per_second', 0):.1f} tok/s  wall {time.time() - start:.0f}s",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="+", help="repeats:depth, e.g. 2000:0.5")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    random.seed(args.seed)
    for case in args.cases:
        repeats, depth = case.split(":")
        ask(args.url, int(repeats), float(depth))


if __name__ == "__main__":
    main()
