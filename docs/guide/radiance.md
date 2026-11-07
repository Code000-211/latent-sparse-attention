# Qwen3.8-Flash-Next's PCA map served by Radiance

D-LSA (Training-Free) for Qwen3.8-Flash-Next served by Radiance on two AMD R9700s at TP2,
with the model's own QSA as the baseline. Everything it needs is in [tools/radiance](../../tools/radiance): the
patches onto Radiance, the dump hook and the scripts that made the map from Radiance's own activations, the start
scripts and the needle test.

This is the served side that [spark.md](spark.md) leaves open. The map here is not the Spark run's: it is fitted from
eight documents Radiance prefilled, not from the extraction corpus, and its `tau` is not calibrated.

## The patches

[tools/radiance/patches](../../tools/radiance/patches) holds seven commits onto Radiance v1.15.1 (`203064b`), as the
branch `lsa-115`:

| patch | what |
|---|---|
| 0001-0005 | LSA from Codeberg PR #414 - the three index ops (`lsa_index_store`, `lsa_cluster`, `lsa_select`) held to the paper's reference, the `--lsa` loader, the ops at decode latency, the map loader's test and `sel_max` for the gated block - ported onto 1.15.1 |
| 0006 | qwen4exp_fp8 at TP2 with sparse prefill, the page lists through the QSA path |
| 0007 | port fixups: ABI 33, `docs/OPS.md`, the ABI test |

PR #414's base, v1.17.0, gives NaN logits on this setup with or without LSA, which is why the port is onto 1.15.1.

```bash
cd ~/Projects/Radiance
git checkout -b lsa-115 203064b
git am /path/to/latent-sparse-attention/tools/radiance/patches/*.patch
cmake --build build-hip
# The ABI changes, so drop the old architecture plugins before they are rebuilt
rm -f build-hip/radiance_home/architectures/*.so
```

## The map

`{L}.W_Q_idx` and `{L}.W_K_idx`, each `[384, 64]` fp32, for the QSA layers 3, 7, ..., 47 - the plugin's flat file.
The `.pt` is an asset of the release
[`d-lsa-pca-qwen38-flash-next-radiance`](https://github.com/Code000-211/latent-sparse-attention/releases/tag/d-lsa-pca-qwen38-flash-next-radiance),
and its sidecar, with the variance kept and the top-256 overlap per layer, is
[lsa_pca_map.json](../../tools/radiance/lsa_pca_map.json).

To make it again:

1. Put [rad_lsa_dump.h](../../tools/radiance/rad_lsa_dump.h) at `arch/common/rad_lsa_dump.h` and call
   `rad_lsa_dump` in the attention block after the query and key norms. The call site is not in the patches - the
   hook syncs the stream once a layer and is for the dump only.
2. Serve without LSA ([start_nolsa.sh](../../tools/radiance/start_nolsa.sh)) with `RADIANCE_LSA_DUMP=<dir>`. Every
   rank appends `r<rank>_L<layer>.bin` per prefill step of at least `RADIANCE_LSA_DUMP_MIN_T` rows (64).
3. [collect.py](../../tools/radiance/collect.py) sends eight documents of Radiance's own source, 32k to 96k tokens,
   through the server.
4. [lsa_pca.py](../../tools/radiance/lsa_pca.py) fits the map as the distillation's PCA initialisation does - rows
   spread evenly over each document, the leading right singular directions of the centred position-free keys, the
   query projection equal to the key projection - joining the ranks' KV heads in rank order:

```bash
python tools/radiance/lsa_pca.py <dump_dir> lsa_pca_map.pt --index-dim 64 --rows 8192 --n-layer 48
```

## Serving

[start_lsa.sh](../../tools/radiance/start_lsa.sh) and [start_nolsa.sh](../../tools/radiance/start_nolsa.sh) read
`RADIANCE` (the build tree), `MODELS` (where the `.rad` checkpoint is), `MAP` and `CTX` (262144 by default).

```bash
tools/radiance/start_lsa.sh
# 2M - the LSA index needs about 3.5 GiB more
CTX=2097152 tools/radiance/start_lsa.sh --host-pool-mib 23552
```

[needle.py](../../tools/radiance/needle.py) hides a passcode at a depth of a filler prompt, one `repeats:depth` a case:

```bash
python tools/radiance/needle.py 500:0.5 2000:0.1 2000:0.5 2000:0.9 8000:0.25 8000:0.75
```

## Results

Two R9700s, TP2, `wht6`, fp8 KV. "The capital of France is" continues "Paris. The capital of Germany is Berlin", and
84*3/2 gives 126.

| context | depth | LSA | LSA prefill tok/s | LSA decode tok/s | QSA | QSA prefill tok/s | QSA decode tok/s |
|---|---|---|---|---|---|---|---|
| 9,689 | 0.50 | OK | 1525 | 78.9 | OK | 1428 | 57.8 |
| 36,089 | 0.10 | OK | 1565 | 81.4 | OK | 1472 | 84.1 |
| 36,089 | 0.50 | OK | 1539 | 104.8 | OK | 1451 | 88.6 |
| 36,089 | 0.90 | OK | 1562 | 127.9 | OK | 1498 | 151.5 |
| 144,089 | 0.25 | OK | 1494 | 98.5 | OK | 1443 | 99.9 |
| 144,089 | 0.75 | OK | 1496 | 142.6 | OK | 1500 | 156.3 |
| 960,089 | 0.50 | OK | 580 | 19.6 | - | - | - |

Prefill with LSA is 2-7 % faster up to 144k. At 960k QSA was not measured.

## Limits

- Pages hold 4 tokens, so a page the selection cuts reads up to 3 keys more than the reference.
- Under MTP speculation the QSA buffers are still allocated for the MTP head.
- `--lsa` is refused at 4 or more ranks, where the KV heads are replicated.
