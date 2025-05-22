#!/bin/bash

# Path to project root (one level up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."

# List of alpha values
ALPHAS=(1)

# List of aligner models
ALIGNER_MODELS=(
  # "/scratch-shared/holy-triangle/weights/img2text_aligner_fixed/flickr30k_cosine/model_best.pth"
  "/scratch-shared/holy-triangle/weights/img2text_aligner_fixed/flickr30k_infonce/model_best.pth"
)

# Inference arguments
PROMPT="a photo of a cat"
REF_IMG="data/cat.jpg"
CONFIG="configs/stable-diffusion/v2-inference-v.yaml"
CKPT="checkpoints/model_checkpoint.ckpt"

# Path to job script
JOB_SCRIPT="${SCRIPT_DIR}/run_single_alpha.sh"

for ALIGNER_MODEL in "${ALIGNER_MODELS[@]}"; do
  for ALPHA in "${ALPHAS[@]}"; do
    echo "Submitting job with alpha=${ALPHA} and model=${ALIGNER_MODEL}"
    sbatch --export=ALL,ALPHA=$ALPHA,PROMPT="$PROMPT",REF_IMG="$REF_IMG",ALIGNER_MODEL="$ALIGNER_MODEL",CONFIG="$CONFIG",CKPT="$CKPT",ROOT_DIR="$ROOT_DIR" "$JOB_SCRIPT"
  done
done
