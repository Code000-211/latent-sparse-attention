#!/usr/bin/env bash
# Step 2, on one Spark: the corpus, collected from the hub and tokenized under Qwen3.8-27B's card.
#
#     SHARED=/mnt/lsa tools/spark/02_data.sh
#     SHARED=/mnt/lsa SOURCES="gutenberg arxiv" tools/spark/02_data.sh     # some sources only
#
# Eight sources stream from the hub, one CPU job each, one after another. `code` is The Stack, which is gated: accept
# its terms on the hub and log in, or leave it out of SOURCES. The two agent-trace sources come from a GitHub release.
# A source already collected is refused by its collector, so a step that stopped half way is run again with the rest.
source "$(dirname "$0")/common.sh"
export HF_HUB_OFFLINE=0

HUB_SOURCES="${SOURCES:-arxiv doab gutenberg hansard usgpo wikimedia swe code}"
for source in $HUB_SOURCES; do
    if [[ -e "$LSA_ROOT/data/sources/$source/_DONE" ]]; then
        echo "$source: collected"
        continue
    fi
    submit_and_wait data prepare "$source"
done

# The traces, rendered and scrubbed, from the release the upstream repository publishes them in
for traces in edgebench games; do
    folder="$LSA_ROOT/data/sources/$traces"
    if compgen -G "$folder/*.parquet" >/dev/null; then
        echo "$traces: downloaded"
        continue
    fi
    mkdir -p "$folder"
    gh release download traces-v1 -R Prefrontal-AI/latent-sparse-attention --pattern "$traces-*" --dir "$folder"
    (cd "$folder" && for f in "$traces"-*; do mv "$f" "${f#"$traces"-}"; done)
done

submit_and_wait data prepare qwen38_27b
