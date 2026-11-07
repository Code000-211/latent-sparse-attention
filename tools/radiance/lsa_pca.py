#!/usr/bin/env python3
"""D-LSA (training-free) map from Radiance dumps (arch/common/rad_lsa_dump.h).

Mirrors lsa.training.distillation: per layer, rows of the key input (the KV heads' position-free dims side
by side, [Hkv * (dh - rot)]) are sampled evenly over each document, INIT_ROWS in total; the map is the
leading right singular directions of the centred rows (model._leading_directions), and the query
projection starts equal to the key projection. Written as the plugin's flat file
`{layer}.W_Q_idx` / `{layer}.W_K_idx` [model_dim, index_dim] fp32, plus a JSON sidecar.

Under TP each rank dumps its own KV heads; the ranks' columns are joined in rank order (rank r holds KV
heads r*Hkv/world ...), giving the full [Hkv_total * (dh - rot)] input. Layers >= --n-layer (MTP) are skipped.

Usage: lsa_pca.py <dump_dir> <out.pt> [--index-dim 64] [--rows 8192] [--n-layer 48]
"""

import argparse
import datetime
import json
import re
from pathlib import Path

import numpy as np
import torch  # type: ignore[import-not-found]  # run with ~/venv/bin/python on dev

MAGIC = 0x4C534131


def records(path: Path):
    """Yield (positions [T], key [T, W], qsum [T, W]) per dumped step."""
    data = np.memmap(path, np.uint8, "r")  # the dumps run to GBs; only the rows read are paged in
    off = 0
    while off + 16 <= len(data):
        magic, T, hkv, fr = (int(v) for v in np.frombuffer(data, np.int32, 4, off))
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic at {off}")
        if off + 16 + 4 * T + 8 * T * hkv * fr > len(data):
            break  # a record still being written
        off += 16
        pos = np.frombuffer(data, np.int32, T, off)
        off += 4 * T
        W = int(hkv * fr)
        key = np.frombuffer(data, np.float32, T * W, off).reshape(T, W)
        off += 4 * T * W
        qs = np.frombuffer(data, np.float32, T * W, off).reshape(T, W)
        off += 4 * T * W
        yield pos, key, qs


def documents(path: Path):
    """Group steps into documents: a step whose first position is 0 starts a new one."""
    doc = []
    for pos, key, qs in records(path):
        if pos[0] == 0 and doc:
            yield doc
            doc = []
        doc.append((pos, key, qs))
    if doc:
        yield doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--index-dim", type=int, default=64)
    ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--n-layer", type=int, default=48)
    a = ap.parse_args()

    by_layer: dict[int, dict[int, Path]] = {}
    for p in a.dump_dir.glob("r*_L*.bin"):
        rank, layer = map(int, re.findall(r"r(\d+)_L(\d+)", p.name)[0])
        if layer < a.n_layer:
            by_layer.setdefault(layer, {})[rank] = p
    if not by_layer:
        raise SystemExit(f"no r*_L*.bin in {a.dump_dir}")
    flat, info = {}, {}
    for layer in sorted(by_layer):
        ranks = by_layer[layer]
        per_rank = [list(documents(ranks[r])) for r in sorted(ranks)]
        docs = []
        for parts in zip(*per_rank, strict=False):
            pos = np.concatenate([p for p, _, _ in parts[0]])
            for other in parts[1:]:
                if not np.array_equal(pos, np.concatenate([p for p, _, _ in other])):
                    raise SystemExit(f"layer {layer}: ranks disagree on positions")
            docs.append((pos, parts))

        def gather(parts, which, rows):
            """Rows `rows` of a document's key (which=1) or qsum (which=2), the ranks' columns joined."""
            cols = []
            for part in parts:
                bounds = np.cumsum([0] + [step[which].shape[0] for step in part])
                out = []
                for i, step in enumerate(part):
                    sel = rows[(rows >= bounds[i]) & (rows < bounds[i + 1])] - bounds[i]
                    out.append(np.asarray(step[which][sel], dtype=np.float32))
                cols.append(np.concatenate(out))
            return np.concatenate(cols, axis=1)

        # INIT_ROWS spread evenly over each document, as model.init_rows does
        per = a.rows // len(docs)
        sampled = []
        for pos, parts in docs:
            n = len(pos)
            sampled.append(gather(parts, 1, np.linspace(0, n - 1, min(per, n)).round().astype(np.int64)))
        rows = torch.from_numpy(np.concatenate(sampled)).float()
        _, s, vh = torch.linalg.svd(rows - rows.mean(0), full_matrices=False)
        proj = vh[: a.index_dim].t().contiguous()
        flat[f"{layer}.W_Q_idx"] = proj.clone()
        flat[f"{layer}.W_K_idx"] = proj.clone()
        var = s**2
        kept = float(var[: a.index_dim].sum() / var.sum())

        # Sanity: how well the 64-dim index ranks keys the way the full position-free score does,
        # on the last document, for the last query rows (causal).
        pos, parts = docs[-1]
        n = len(pos)
        k = torch.from_numpy(gather(parts, 1, np.arange(n)))
        q = torch.from_numpy(gather(parts, 2, np.arange(n - 64, n)))
        full = q @ k.t()
        ik = torch.nn.functional.normalize(k @ proj, dim=-1)
        iq = torch.nn.functional.normalize(q @ proj, dim=-1)
        idx = iq @ ik.t()
        top = min(256, k.shape[0] // 4)
        a_top = full.topk(top, dim=-1).indices
        b_top = idx.topk(top, dim=-1).indices
        overlap = float(
            np.mean(
                [len(set(x.tolist()) & set(y.tolist())) / top for x, y in zip(a_top, b_top, strict=False)]
            )
        )
        info[layer] = {
            "docs": len(docs),
            "rows": int(rows.shape[0]),
            "dim": int(rows.shape[1]),
            "variance_kept": round(kept, 4),
            f"top{top}_overlap": round(overlap, 3),
        }
        print(
            f"layer {layer:2d}: {len(docs)} docs, {rows.shape[0]} rows x {rows.shape[1]}, "
            f"variance in {a.index_dim} dims {kept:.3f}, top-{top} overlap vs position-free score {overlap:.3f}",
            flush=True,
        )

    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(flat, a.out)
    side = {
        "input": "key_and_query",
        "run": "radiance-dump-pca",
        "checkpoint": "step0 (PCA init)",
        "date": datetime.date.today().isoformat(),
        "note": "D-LSA training-free: PCA of the position-free key heads, dumped by Radiance "
        "(qwen3.8-next-flash-fp8-iq4r-moe, YaRN x9). tau not calibrated.",
        "layers": info,
    }
    a.out.with_suffix(".json").write_text(json.dumps(side, indent=2))
    print(f"{a.out}: layers {sorted(info)}, each map {tuple(next(iter(flat.values())).shape)}")


if __name__ == "__main__":
    main()
