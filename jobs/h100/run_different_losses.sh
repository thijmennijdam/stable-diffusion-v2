#!/bin/bash

declare -a VERSIONS=("v1" "v2")
declare -a DATASETS=("coco" "flickr30k")
declare -a LOSSES=("infonce" "mse" "mmd") # "mse", "mmd"
declare -a BATCH_SIZES=("64") #"128")
declare -a CLS=("true")
declare -a DROPOUT=("0.1")

for VERSION in "${VERSIONS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for LOSS in "${LOSSES[@]}"; do
      for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        for DROP in "${DROPOUT[@]}"; do
          for CL in "${CLS[@]}"; do
            sbatch --export=ALL,VERSION=$VERSION,DATASET=$DATASET,LOSS=$LOSS,BATCH_SIZE=$BATCH_SIZE,DROPOUT=$DROP,CL=$CL ./jobs/h100/run_loss.job
          done
        done
      done
    done
  done
done
