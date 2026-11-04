#!/usr/bin/env bash
# Step 3, on every Spark at once: one stage of the teacher each.
#
#     SHARED=/mnt/lsa tools/spark/03_extract.sh 0     # on spark 1
#     SHARED=/mnt/lsa tools/spark/03_extract.sh 1     # on spark 2
#     SHARED=/mnt/lsa tools/spark/03_extract.sh 2     # on spark 3
#     SHARED=/mnt/lsa tools/spark/03_extract.sh 3     # on spark 4
#
# Stage 0 holds the embedding, layers 0-3 and the 51B n-gram table on layer 1 - about 114 GB, the tightest of the four
# - so give it the node with the least else running. A stage after the first takes a document once the stage before it
# has published that document's streams, so all four can start together and the later ones wait.
source "$(dirname "$0")/common.sh"

stage="${1:?usage: 03_extract.sh STAGE (0-3)}"
lsa distill targets "$EXTRACTION" --stage "$stage" --dry-run
submit_and_wait distill targets "$EXTRACTION" --stage "$stage"
