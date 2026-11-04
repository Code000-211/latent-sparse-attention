# Sourced by every step in tools/spark - where things live and how a step waits for the job it submitted.
#
# SHARED is the directory every Spark mounts at the same path. Everything large lives under it: the checkpoint (some
# 360 GB), the corpus, the extraction cache and the boundary streams the stages hand each other.

set -euo pipefail

SHARED="${SHARED:-/mnt/lsa}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export LSA_ROOT="${LSA_ROOT:-$SHARED/runs}"
export HF_HOME="${HF_HOME:-$SHARED/hf}"
export LSA_SITE=spark

EXTRACTION="configs/data/targets/qwen38_flash_next/long_128k_2m_key_and_query.yaml"
TRAINING="configs/training/distillation/qwen38_flash_next/long_128k_2m_key_and_query_pca.yaml"
RUN_DIR="$LSA_ROOT/training/distillation/qwen38_flash_next/long_128k_2m_key_and_query_pca"
MAP="$LSA_ROOT/models/maps/qwen38_flash_next__long-128k-2M-key-and-query-pca-step0.pt"

FLASH_NEXT=Qwen/Qwen3.8-Flash-Next
FLASH_NEXT_REVISION=de4b8e4d43b917e7706784d8bb445c9af86a3540
# The corpus is tokenized under Qwen3.8-27B's card, whose tokenizer files are Flash-Next's byte for byte
QWEN27B=Qwen/Qwen3.8-27b
QWEN27B_REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

cd "$REPO"

lsa() {
    uv run --no-sync lsa --site spark "$@"
}

# Run an `lsa` command that submits one local job, then block until that job has finished.
# A local job is detached: it outlives the command that submitted it, so the step waits on submitit's result file.
submit_and_wait() {
    local out job result
    out="$(lsa "$@")"
    echo "$out"
    job="$(sed -n 's/^submitted \([^ ]*\).*/\1/p' <<<"$out" | tail -1)"
    if [[ -z "$job" ]]; then
        echo "no job was submitted" >&2
        return 1
    fi
    echo "waiting for job $job - logs under $LSA_ROOT/logs"
    while :; do
        result="$(find "$LSA_ROOT/logs" -name "${job}_0_result.pkl" -print -quit 2>/dev/null)"
        [[ -n "$result" ]] && break
        sleep 30
    done
    if ! uv run --no-sync python - "$result" <<'PY'
import pickle, sys
outcome, value = pickle.load(open(sys.argv[1], "rb"))
if outcome != "success":
    print(value, file=sys.stderr)
    sys.exit(1)
PY
    then
        echo "job $job failed - its log is beside $result" >&2
        return 1
    fi
    echo "job $job finished"
}
