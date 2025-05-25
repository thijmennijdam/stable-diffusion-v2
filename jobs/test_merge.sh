#!/bin/bash

# Path to project root (one level up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."

# Experiment settings
ALPHAS=(0.3)
ALIGNER_MODELS=(
  "/scratch-shared/holy-triangle/weights/img2text_aligner_fixed/flickr30k_infonce/model_best.pth"
)
FUSION_TOKEN_TYPES=("all")
FUSION_TYPES=("alpha_blend" "cross_attention" "concat")
USE_PNO_OPTIONS=("false")  # Add PNO and non-PNO variants

# Inference arguments
PROMPT="a photo of a cat"
REF_IMG="data/van_gogh_starry_night.jpg"
CONFIG="configs/stable-diffusion/v2-inference-v.yaml"
CKPT="/scratch-shared/holy-triangle/weights/stable-diffusion-2-1/v2-1_768-ema-pruned.ckpt"
JOB_SCRIPT="${SCRIPT_DIR}/test_merge_single_run.sh"

# Create logs directory if it doesn't exist
mkdir -p "${SCRIPT_DIR}/../logs"

echo "Starting experiment with:"
echo "  - Alpha values: ${ALPHAS[*]}"
echo "  - Fusion token types: ${FUSION_TOKEN_TYPES[*]}"
echo "  - Fusion methods: ${FUSION_TYPES[*]}"
echo "  - PNO modes: ${USE_PNO_OPTIONS[*]}"
echo "  - Total jobs: $((${#ALPHAS[@]} * ${#ALIGNER_MODELS[@]} * ${#FUSION_TOKEN_TYPES[@]} * ${#FUSION_TYPES[@]} * ${#USE_PNO_OPTIONS[@]}))"
echo ""

for USE_PNO in "${USE_PNO_OPTIONS[@]}"; do
  for ALIGNER_MODEL in "${ALIGNER_MODELS[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
      for FUSION_TOKEN_TYPE in "${FUSION_TOKEN_TYPES[@]}"; do
        for FUSION_TYPE in "${FUSION_TYPES[@]}"; do
          echo "Submitting job with alpha=${ALPHA}, model=$(basename "$ALIGNER_MODEL"), token_type=${FUSION_TOKEN_TYPE}, fusion_type=${FUSION_TYPE}, PNO=${USE_PNO}"
          sbatch --export=ALL,ALPHA=$ALPHA,PROMPT="$PROMPT",REF_IMG="$REF_IMG",ALIGNER_MODEL="$ALIGNER_MODEL",CONFIG="$CONFIG",CKPT="$CKPT",ROOT_DIR="$ROOT_DIR",FUSION_TOKEN_TYPE="$FUSION_TOKEN_TYPE",FUSION_TYPE="$FUSION_TYPE",USE_PNO="$USE_PNO" "$JOB_SCRIPT"
        done
      done
    done
  done
done
