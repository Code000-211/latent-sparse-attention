#!/usr/bin/env bash
# Step 1, on one Spark: the checkpoint and the tokenizer, into the shared hub cache.
#
#     SHARED=/mnt/lsa tools/spark/01_download.sh
#
# Flash-Next is about 360 GB. Log in first (`uv run --no-sync hf auth login`) if the hub asks for it.
source "$(dirname "$0")/common.sh"
export HF_HUB_OFFLINE=0

uv run --no-sync hf download "$FLASH_NEXT" --revision "$FLASH_NEXT_REVISION"
uv run --no-sync hf download "$QWEN27B" --revision "$QWEN27B_REVISION" \
    --include "tokenizer*" "vocab.json" "merges.txt" "chat_template.jinja" "config.json" "generation_config.json"

uv run --no-sync python - <<'PY'
from lsa.evaluation.adapter import adapter_for
from lsa.evaluation.config import ModelConfig

config = ModelConfig(name="qwen38_flash_next")
adapter = adapter_for(config)
layers = adapter.full_attention_layers(adapter.text_config(config), config)
print(f"{type(adapter).__name__}, QSA layers {layers}")
assert layers == list(range(3, 48, 4)), layers
PY
