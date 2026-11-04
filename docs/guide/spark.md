# Qwen3.8-Flash-Next's PCA map on four DGX Sparks

How to make D-LSA (Training-Free) for Qwen3.8-Flash-Next - the map, and its step-0 report against QSA - on four DGX
Sparks. Each step is one script in [tools/spark](../../tools/spark), and every script reads the same `SHARED`.

## What it needs

- **Four DGX Sparks** (GB10, sm_121, aarch64, 128 GB each), with this branch checked out at the same path on each.
- **One filesystem every Spark mounts at the same path**, given as `SHARED`. It holds the checkpoint (about 360 GB),
  the corpus, the extraction cache (about 0.4 TB) and the streams the stages hand each other (up to about 40 GB per 2M
  document in flight). The stages exchange those streams through it, so its bandwidth paces the extraction.
- **uv**, and a Hugging Face login for the downloads. `code` (The Stack) is gated: accept its terms, or leave it out.
- **gh**, logged in, for the agent traces' release.

The site is [configs/site/spark.yaml](../../configs/site/spark.yaml): `launcher: local`, so every job runs on the
node that submits it and nothing spans nodes. The scripts set `LSA_ROOT=$SHARED/runs` and `HF_HOME=$SHARED/hf`.

## The steps

| step | where | what | roughly |
|---|---|---|---|
| `00_setup.sh` | every Spark | `uv sync` (evaluation and training extras, without causal-conv1d), checks the GPU, the adapter and that `SHARED` is writable | minutes |
| `01_download.sh` | one Spark | Flash-Next at its pinned revision, and Qwen3.8-27B's tokenizer files | the 360 GB download |
| `02_data.sh` | one Spark | collects the eight hub sources, downloads the two trace sources, tokenizes all ten under `qwen38_27b` | hours to days, CPU |
| `03_extract.sh N` | Spark N+1, all four together | stage N of the teacher: spans [0, 4), [4, 20), [20, 34), [34, 48) | the long one |
| `04_pca.sh` | one Spark, after all four stages | fits, reports and exports the map | under an hour |

```bash
export SHARED=/mnt/lsa
tools/spark/00_setup.sh
```

The setup step runs on all four Sparks. Steps 1 and 2 run on one of them.

```bash
tools/spark/01_download.sh
```

```bash
tools/spark/02_data.sh
```

Step 3 then runs on all four at once, one stage each - stage 0 on the first Spark, 1 on the second, and so on.

```bash
tools/spark/03_extract.sh 0
```

Step 4 runs on one Spark once every stage has finished.

```bash
tools/spark/04_pca.sh
```

The map lands at `$SHARED/runs/models/maps/qwen38_flash_next__long-128k-2M-key-and-query-pca-step0.pt`, and the
report at `$SHARED/runs/training/distillation/qwen38_flash_next/long_128k_2m_key_and_query_pca/report_step0.json`.

A step that submits a job waits for it and fails with its traceback. Logs are under `$SHARED/runs/logs`, one folder a job.

## What to check on the way

- **After step 2:** `$SHARED/runs/data/tokenized/qwen38_27b/` holds a `.bin` and a `.manifest.parquet` per source.
- **At the start of step 3:** the extraction writes `selection.json` under its output before it loads a weight. It
  lists every window by length. If the 2M bucket is empty, no source document reaches 2,097,152 tokens, and the report
  has no 2M row. The `2m` evaluation set in the training config then has nothing to read and has to be dropped.
- **Stage 0's memory:** the embedding, layers 0-3 and layer 1's 51B-parameter n-gram table are about 114 GB of the
  GB10's 128. If it runs out, lower `run.prefill` (16384 in the extraction config) or stop everything else on that node.
  If it still does not fit, the n-gram table has to come off the card, which these configs do not do yet.
- **Stages 1-3** wait for stage 0's streams of each document, so they idle at first. That is expected.

## Reading the report

`report_step0.json` scores the untrained (PCA) map per held-out set - 128k, 512k, 1M and 2M - by how much of the
teacher's attention its selection keeps and how many of the teacher's strongest keys it finds. These are offline
numbers, and QSA is not among them: the extraction keeps QSA's own picks only with `data.queries.native: true`, which
this config leaves off. The served comparison - RULER
and PaperTrail at the same lengths, against QSA - needs the map served, which is the work planned in RadianceSwap's
`docs/LSA.md`.

## What is untested

Nothing here has run on a Spark yet - the adapter is checked on CPU against the transformers module at a small size,
and the scripts by dry runs. The first run is the test of: aarch64 wheels on the Spark, Triton on sm_121, stage 0's
memory, and YaRN x9 (the model card's rotation) at 2M.
