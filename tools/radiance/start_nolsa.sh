#!/bin/sh
# Qwen3.8-Flash-Next served by Radiance on the model's own QSA - the baseline of start_lsa.sh, same flags without --lsa.
RADIANCE=${RADIANCE:-$HOME/Projects/Radiance-lsa/build-hip}
MODELS=${MODELS:-$HOME/models}
CTX=${CTX:-262144}
cd "$MODELS" || exit 1
RADIANCE_HOME=$RADIANCE/radiance_home exec "$RADIANCE/bin/radiance" --model qwen3.8-next-flash-fp8-iq4r-moe.rad \
  --tp 2 --tp-wire wht6 --rope-scaling yarn:9 --max-model-len "$CTX" --max-num-seqs 1 \
  --kv-cache-dtype fp8 --placement expert_tiered --host-pool-mib 21504 --weights-disk-tier --gpu-headroom-mib 96 \
  --max-num-batched-tokens 1024 --debug-placement --host 127.0.0.1 --port 8000 "$@"
