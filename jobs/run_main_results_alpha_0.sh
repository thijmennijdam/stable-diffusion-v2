#!/bin/bash

# Path to project root (one level up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."

# Small set of alpha values for testing
ALPHAS=(0)

# Best aligner model based on main_results.sh settings
ALIGNER_MODEL="${ROOT_DIR}/weights/aligner_models/version_v1/dataset_coco/loss_combined/batch_64/model_best.pth"

# Fusion token type (using "all" as in main_results.sh)
FUSION_TOKEN_TYPE="all"

# All fusion types from main_results.sh
FUSION_TYPES=("alpha_blend")

All reference images from data folder
REF_IMAGES=(
  "data/texture.jpg"
  "data/doggo.jpg" 
  "data/mattise.jpg"
  "data/totoro.jpg"
  "data/pearl_earing.jpg"
  "data/ghibli.jpg"
  "data/van_gogh.jpg"
  "data/picasso.jpg"
  "data/surreal.png"
  "data/pop_art.png"
  "data/the-persistence-of-memory-dali.jpg"
  "data/the-wave-hokusai.jpg"
  "data/the-scream.jpg"
  "data/cat.jpg"
  "data/van_gogh_starry_night.jpg"
  "data/sketch_penguin.jpg"
  "data/picasso_style.jpg"
)

# Inference arguments (updated from main_results.sh)
PROMPT="a photo of a cat"
CONFIG="configs/stable-diffusion/v2-inference-v.yaml"
CKPT="./weights/v2-1_768-ema-pruned.ckpt"

# Path to job script
JOB_SCRIPT="${SCRIPT_DIR}/run_single_alpha_fuse.sh"

# Create logs directory if it doesn't exist
mkdir -p "${SCRIPT_DIR}/../logs"

echo "Starting main results experiment with:"
echo "  - Alpha values: ${ALPHAS[*]}"
echo "  - Aligner model: $(basename "$ALIGNER_MODEL")"
echo "  - Fusion token type: ${FUSION_TOKEN_TYPE}"
echo "  - Fusion methods: ${FUSION_TYPES[*]}"
echo "  - Reference images: ${#REF_IMAGES[@]} images"
echo "  - Total jobs: $((${#ALPHAS[@]} * ${#FUSION_TYPES[@]} * ${#REF_IMAGES[@]}))"
echo ""

# Run conditioned experiments (alpha > 0)
echo "=== Submitting experiments ==="
for ALPHA in "${ALPHAS[@]}"; do
  for FUSION_TYPE in "${FUSION_TYPES[@]}"; do
    for REF_IMG in "${REF_IMAGES[@]}"; do
      REF_IMG_NAME=$(basename "$REF_IMG" | cut -d. -f1)
      echo "Submitting job: alpha=${ALPHA}, fusion=${FUSION_TYPE}, ref=${REF_IMG_NAME}"
      sbatch --export=ALL,ALPHA=$ALPHA,PROMPT="$PROMPT",REF_IMG="$REF_IMG",ALIGNER_MODEL="$ALIGNER_MODEL",CONFIG="$CONFIG",CKPT="$CKPT",ROOT_DIR="$ROOT_DIR",FUSION_TOKEN_TYPE="$FUSION_TOKEN_TYPE",FUSION_TYPE="$FUSION_TYPE" "$JOB_SCRIPT"
    done
  done
done

echo ""
echo "All jobs submitted successfully!"
echo "Total experiments: $((${#ALPHAS[@]} * ${#FUSION_TYPES[@]} * ${#REF_IMAGES[@]})) jobs"
echo "Monitor with: squeue -u \$USER" 