#!/bin/bash

declare -a LOSSES=("infonce")
declare -a BATCH_SIZES=("64")
declare -a WEIGHT_INIT=("default")
declare -a DROPOUT=("0.1")

for LOSS in "${LOSSES[@]}"; do
  for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    for WEIGHT in "${WEIGHT_INIT[@]}"; do
      for DROP in "${DROPOUT[@]}"; do
        MODEL_PATH="weights/img2text_alignerv1/flickr30k/${LOSS}/${BATCH_SIZE}/model.pth"
        sbatch --export=ALL,LOSS=$LOSS,BATCH_SIZE=$BATCH_SIZE,MODEL_PATH=$MODEL_PATH,WEIGHT_INIT=$WEIGHT,DROPOUT=$DROP ./jobs/h100/run_lossv2.job
      done
    done
  done
done
