#!/bin/bash

# Path to project root (one level up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."

# List of alpha values
ALPHAS=(1)

# List of aligner models
ALIGNER_MODELS=(
#   "/scratch-shared/holy-triangle/weights/img2text_aligner_fixed/flickr30k_cosine/model_best.pth"
  "/scratch-shared/holy-triangle/weights/img2text_aligner_fixed/flickr30k_infonce/model_best.pth"
)

# List of fusion token types (leveraging existing exclude_cls parameter)
FUSION_TOKEN_TYPES=("cls_only" "except_cls" "all")

# List of ref_first options
REF_FIRST_OPTIONS=("true" "false")

# Inference arguments
PROMPT="a photo of a cat"
REF_IMG="data/cat.jpg"
CONFIG="configs/stable-diffusion/v2-inference-v.yaml"
CKPT="model_checkpoint.ckpt"

# Path to job script
JOB_SCRIPT="${SCRIPT_DIR}/run_single_alpha.sh"

for ALIGNER_MODEL in "${ALIGNER_MODELS[@]}"; do
  for ALPHA in "${ALPHAS[@]}"; do
    for FUSION_TOKEN_TYPE in "${FUSION_TOKEN_TYPES[@]}"; do
      for REF_FIRST in "${REF_FIRST_OPTIONS[@]}"; do
        echo "Submitting job with alpha=${ALPHA}, model=${ALIGNER_MODEL}, fusion_token_type=${FUSION_TOKEN_TYPE}, ref_first=${REF_FIRST}"
        sbatch --export=ALL,ALPHA=$ALPHA,PROMPT="$PROMPT",REF_IMG="$REF_IMG",ALIGNER_MODEL="$ALIGNER_MODEL",CONFIG="$CONFIG",CKPT="$CKPT",ROOT_DIR="$ROOT_DIR",FUSION_TOKEN_TYPE="$FUSION_TOKEN_TYPE",REF_FIRST="$REF_FIRST" "$JOB_SCRIPT"
      done
    done
  done
done