#!/bin/bash

declare -a LOSSES=("infonce")
declare -a BATCH_SIZES=("64" "128")


for LOSS in "${LOSSES[@]}"; do
  for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    MODEL_PATH="weights/img2text_alignerv1/flickr30k/${LOSS}/${BATCH_SIZE}/model.pth"
    sbatch --export=ALL,LOSS=$LOSS,BATCH_SIZE=$BATCH_SIZE,MODEL_PATH=$MODEL_PATH ./jobs/h100/run_loss.job
  done
done
