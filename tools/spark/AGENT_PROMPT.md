# Task: build the training-free LSA map (PCA) for Qwen3.8-Flash-Next on 4 DGX Sparks

You are a coding agent with shell access to four NVIDIA DGX Sparks. Your job is to use this repository
(`latent-sparse-attention`, delivered with this prompt as a ZIP) to build the **training-free PCA map of Latent
Sparse Attention (D-LSA) for Qwen3.8-Flash-Next**, together with its offline report. You do not train a model: the map
is a principal component analysis of the keys of the frozen model.

The authoritative guide is `docs/guide/spark.md` in the repository. Read it in full first, then read
`configs/site/spark.yaml`, `tools/spark/*.sh`,
`configs/data/targets/qwen38_flash_next/long_128k_2m_key_and_query.yaml` and
`configs/training/distillation/qwen38_flash_next/long_128k_2m_key_and_query_pca.yaml`.

## The environment you will find or have to set up

- Four DGX Sparks: GB10, compute capability sm_121, aarch64, 128 GB of unified memory each.
- The repository unpacked at **the same path** on every Spark.
- One shared filesystem that all four mount at **the same path**. You are given this path as `SHARED`. If you do
  not know it, ask before you start anything.
- `uv`, a Hugging Face login (`uv run --no-sync hf auth login`) and a logged-in `gh`.

## Procedure

On every Spark, set `export SHARED=<path>`, then work through the steps **in this order**:

1. **On all four Sparks:** `tools/spark/00_setup.sh`. Each one must print:
   - `NVIDIA GB10, sm_121, ...`;
   - `datacenter Blackwell: False`;
   - `ok` at the end;
   - all four hostnames in the node list.
2. **On one Spark:** `tools/spark/01_download.sh`. This is about 360 GB. It must end with
   `QSA layers [3, 7, 11, ..., 47]`.
3. **On one Spark:** `tools/spark/02_data.sh`.
   - **Duration:** hours to days, CPU only.
   - **`code` (The Stack):** this source is gated. If access has not been granted, set
     `SOURCES="arxiv doab gutenberg hansard usgpo wikimedia swe"` and report it.
   - **Interruptions:** the script can be rerun, and sources that are already done are skipped.
4. **On all four Sparks at the same time:** `tools/spark/03_extract.sh N`. Spark 1 runs stage `0`, Spark 2 runs
   stage `1`, and so on.
   - **Memory:** stage 0 is the memory-critical one (~114 GB of 128). Give it the Spark that runs nothing else.
   - **Waiting:** stages 1 to 3 wait for stage 0 at first. That is expected.
5. **On one Spark, once all four stages have finished:** `tools/spark/04_pca.sh`.

Every script that submits a job waits for that job to end. If the job fails, the script exits with the job's
traceback. Logs are under `$SHARED/runs/logs/`.

## Checkpoints where you stop and report

- **After step 3:** `$SHARED/runs/data/tokenized/qwen38_27b/` holds one `.bin` and one `.manifest.parquet` per source.
- **Right after step 4 starts:**
  - **What you read:** `$SHARED/runs/data/targets/qwen38_flash_next/long_128k_2m_key_and_query/qwen38_flash_next/selection.json`
    (it is in the collection's folder). Count the windows per length (131072, 524288, 1048576, 2097152).
  - **If there are no 2097152 windows:** stop and report it.
  - **What has to happen next:** the `2m` entry under `data.evaluate` must then be removed from the training config.
    Change the config only after asking.
- **On a failure in step 4 or 5:** stop and report the traceback. Do not try to configure your way around it.

## Known risks and what to do

- **Out of memory in stage 0:**
  - Lower `run.prefill` in the extraction config from 16384 to 8192 and restart the stage. Documents already written
    are skipped.
  - If that is not enough, stop and report. The n-gram table (layer 1, ~102 GB) would then have to come off the GPU,
    and that is not implemented here.
- **A package fails to build in `uv sync` on aarch64:** report the package and the error. Do not drop dependencies,
  except `causal-conv1d`, which the script already leaves out.
- **Triton or CUDA errors on sm_121:** report them verbatim, with the kernel name and the stack trace.
- **A download breaks off:** rerun it. `hf download` resumes.

## Rules

- **Delete nothing**, above all nothing under `$SHARED`, in the Hugging Face cache, or in any folder you did not
  create yourself. If you need space, ask.
- **No code changes** under `src/`. Change configs only as far as described above, and report every change.
- **Do not push**, and open no pull requests.
- **Report honestly:** if something did not run or you skipped it, say so. Do not give any number you did not read
  from a file or an output.

## What you hand back at the end

1. **The path of the map:** `$SHARED/runs/models/maps/qwen38_flash_next__long-128k-2M-key-and-query-pca-step0.pt`.
2. **The contents of `report_step0.json`** from
   `$SHARED/runs/training/distillation/qwen38_flash_next/long_128k_2m_key_and_query_pca/`, with the map's metrics for
   each length.
3. **The window counts per length** from `selection.json`.
4. **Run times:** how long each stage and step 5 took, and the peak memory of stage 0 if the logs show it.
5. **Deviations:** every deviation from these instructions, every config you changed, and every error that occurred.
