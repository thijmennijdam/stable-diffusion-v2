#!/bin/bash

declare -a VERSIONS=("v1")
declare -a DATASETS=("coco")
# declare -a LOSSES=("infonce" "cross_attention" "combined")
declare -a LOSSES=("combined")

for VERSION in "${VERSIONS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for LOSS in "${LOSSES[@]}"; do
      sbatch --export=ALL,VERSION=$VERSION,DATASET=$DATASET,LOSS=$LOSS ./jobs/h100/train_aligners/train_aligner_single.job
    done
  done
done
