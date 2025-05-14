#!/bin/bash

# Path to project root (one level up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."

# List of alpha values
ALPHAS=(0 0.01 0.05 0.1 0.15 0.2)

# Inference arguments
PROMPT="a photo of a cat"
REF_IMG="data/cat.jpg"
ALIGNER_MODEL="model_best_cosine_flickr30k.pth"
CONFIG="configs/stable-diffusion/v2-inference-v.yaml"
CKPT="model_checkpoint.ckpt"

# Path to job script
JOB_SCRIPT="${SCRIPT_DIR}/run_single_alpha.sh"

for ALPHA in "${ALPHAS[@]}"; do
  echo "Submitting job with alpha=${ALPHA}"
  sbatch --export=ALL,ALPHA=$ALPHA,PROMPT="$PROMPT",REF_IMG="$REF_IMG",ALIGNER_MODEL="$ALIGNER_MODEL",CONFIG="$CONFIG",CKPT="$CKPT",ROOT_DIR="$ROOT_DIR" "$JOB_SCRIPT"
done
