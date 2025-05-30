#!/bin/bash

# Path to project root (one level up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."

# List of alpha values
# ALPHAS=(0.1 0.15 0.3 0.5)
ALPHAS=(0.3)

# List of aligner models
ALIGNER_MODELS=(
#   "/scratch-shared/holy-triangle/weights/img2text_aligner_fixed/flickr30k_cosine/model_best.pth"
  "/scratch-shared/holy-triangle/weights/img2text_aligner_fixed/flickr30k_infonce/model_best.pth"
)

# List of fusion token types
# FUSION_TOKEN_TYPES=("cls_only" "except_cls" "all")
FUSION_TOKEN_TYPES=("all")

# List of fusion methods
# FUSION_TYPES=("concat" "cross_attention")
FUSION_TYPES=("alpha_blend" "cross_attention" "concat")

# Inference arguments
PROMPT="a photo of a cat"
REF_IMG="data/sketch_penguin.jpg"
CONFIG="configs/stable-diffusion/v2-inference-v.yaml"
CKPT="/scratch-shared/holy-triangle/weights/stable-diffusion-2-1/v2-1_768-ema-pruned.ckpt"

# Path to job script
JOB_SCRIPT="${SCRIPT_DIR}/run_single_alpha_fuse.sh"

# Create logs directory if it doesn't exist
mkdir -p "${SCRIPT_DIR}/../logs"

echo "Starting experiment with:"
echo "  - Alpha values: ${ALPHAS[*]}"
echo "  - Fusion token types: ${FUSION_TOKEN_TYPES[*]}"
echo "  - Fusion methods: ${FUSION_TYPES[*]}"
echo "  - Total jobs: $((${#ALPHAS[@]} * ${#ALIGNER_MODELS[@]} * ${#FUSION_TOKEN_TYPES[@]} * ${#FUSION_TYPES[@]}))"
echo ""

for ALIGNER_MODEL in "${ALIGNER_MODELS[@]}"; do
  for ALPHA in "${ALPHAS[@]}"; do
    for FUSION_TOKEN_TYPE in "${FUSION_TOKEN_TYPES[@]}"; do
      for FUSION_TYPE in "${FUSION_TYPES[@]}"; do
        echo "Submitting job with alpha=${ALPHA}, model=$(basename "$ALIGNER_MODEL"), fusion_token_type=${FUSION_TOKEN_TYPE}, FUSION_TYPE=${FUSION_TYPE}"
        sbatch --export=ALL,ALPHA=$ALPHA,PROMPT="$PROMPT",REF_IMG="$REF_IMG",ALIGNER_MODEL="$ALIGNER_MODEL",CONFIG="$CONFIG",CKPT="$CKPT",ROOT_DIR="$ROOT_DIR",FUSION_TOKEN_TYPE="$FUSION_TOKEN_TYPE",FUSION_TYPE="$FUSION_TYPE" "$JOB_SCRIPT"
      done
    done
  done
done