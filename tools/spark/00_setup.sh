#!/usr/bin/env bash
# Step 0, on every Spark: the environment, and the checks that say this node can take part.
#
#     SHARED=/mnt/lsa tools/spark/00_setup.sh
source "$(dirname "$0")/common.sh"

[[ "$(uname -m)" == "aarch64" ]] || echo "warning: $(uname -m), not the Spark's aarch64"
command -v uv >/dev/null || { echo "uv is missing - curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }

# causal-conv1d builds from source and only GLM's delta layers use it - Flash-Next's convolution falls back to torch.
# The flash-attn wheels are x86_64 only and are skipped by their markers on aarch64
uv sync --locked --extra evaluation --extra training --no-install-package causal-conv1d

mkdir -p "$LSA_ROOT" "$HF_HOME"
marker="$SHARED/.node-$(hostname)"
date > "$marker" || { echo "$SHARED is not writable here" >&2; exit 1; }
echo "nodes that have written to $SHARED so far:"
ls -1 "$SHARED"/.node-* | sed 's/.*\.node-/  /'

uv run --no-sync python - <<'PY'
import torch
from lsa.core.kernels import dispatch
from lsa.evaluation.adapter import ADAPTERS

assert torch.cuda.is_available(), "no CUDA device"
device = torch.device("cuda", 0)
name, capability = torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)
free, total = torch.cuda.mem_get_info()
print(f"{name}, sm_{capability[0]}{capability[1]}, {free / 2**30:.0f} of {total / 2**30:.0f} GiB free")
print(f"datacenter Blackwell: {dispatch.datacenter_blackwell(device)} (a GB10 must say False)")
assert "qwen4_exp" in ADAPTERS, "the qwen4_exp adapter is not registered - is this branch checked out?"
x = torch.randn(256, 256, device=device, dtype=torch.bfloat16)
torch.testing.assert_close((x @ x).float(), x.float() @ x.float(), atol=1.0, rtol=0.05)
print("ok")
PY
