#!/bin/bash

# Path to project root (one level up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."

# List of alpha values
# ALPHAS=(0 0.01 0.05 0.1 0.15 0.2)
# ALPHAS=(0 0.1 0.3 0.5 0.8)
ALPHAS=(0.1 0.3 0.5)

# Aligner model parameters
declare -a ALIGNER_VERSIONS=("v1") #"v2")
declare -a ALIGNER_DATASETS=("coco") #"flickr30k")
declare -a ALIGNER_LOSSES=("mse") #"infonce" "mse" "mmd")
declare -a ALIGNER_BATCH_SIZES=("64") #"128")
declare -a ALIGNER_DROPOUT=("0.1")
declare -a ALIGNER_CLS=("true")

# Inference arguments
PROMPT="a photo of a cat"
REF_IMG="data/cat.jpg"
CONFIG="configs/stable-diffusion/v2-inference-v.yaml"
CKPT="model_checkpoint.ckpt"

# Path to job script
JOB_SCRIPT="${SCRIPT_DIR}/run_single_alpha.sh"

for ALIGNER_VERSION in "${ALIGNER_VERSIONS[@]}"; do
  for ALIGNER_DATASET in "${ALIGNER_DATASETS[@]}"; do
    for ALIGNER_LOSS in "${ALIGNER_LOSSES[@]}"; do
      for ALIGNER_BATCH_SIZE in "${ALIGNER_BATCH_SIZES[@]}"; do
        for ALIGNER_DROPOUT_VAL in "${ALIGNER_DROPOUT[@]}"; do
          for ALIGNER_CLS_VAL in "${ALIGNER_CLS[@]}"; do
            for ALPHA in "${ALPHAS[@]}"; do
              echo "Submitting job with alpha=${ALPHA}, aligner_version=${ALIGNER_VERSION}, dataset=${ALIGNER_DATASET}, loss=${ALIGNER_LOSS}, batch_size=${ALIGNER_BATCH_SIZE}, dropout=${ALIGNER_DROPOUT_VAL}, exclude_cls=${ALIGNER_CLS_VAL}"
              sbatch --export=ALL,ALPHA=$ALPHA,PROMPT="$PROMPT",REF_IMG="$REF_IMG",CONFIG="$CONFIG",CKPT="$CKPT",ROOT_DIR="$ROOT_DIR",ALIGNER_VERSION=$ALIGNER_VERSION,ALIGNER_DATASET=$ALIGNER_DATASET,ALIGNER_LOSS=$ALIGNER_LOSS,ALIGNER_BATCH_SIZE=$ALIGNER_BATCH_SIZE,ALIGNER_DROPOUT=$ALIGNER_DROPOUT_VAL,ALIGNER_CLS=$ALIGNER_CLS_VAL "$JOB_SCRIPT"
            done
          done
        done
      done
    done
  done
done
