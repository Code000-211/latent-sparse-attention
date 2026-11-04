#!/usr/bin/env bash
# Step 4, on one Spark once every stage of step 3 has finished: fit the PCA map, report it, export it.
#
#     SHARED=/mnt/lsa tools/spark/04_pca.sh
#
# A run of zero steps fits each QSA layer's map to the top 64 principal directions of its key input, calibrates the
# temperature, and scores the untrained map on the held-out sets - the step-0 report is the offline read against QSA.
source "$(dirname "$0")/common.sh"

submit_and_wait distill train "$TRAINING"

mkdir -p "$(dirname "$MAP")"
uv run --no-sync lsa --site spark distill export "$RUN_DIR" "$MAP"
echo
echo "map:    $MAP"
echo "report: $RUN_DIR/report_step0.json"
