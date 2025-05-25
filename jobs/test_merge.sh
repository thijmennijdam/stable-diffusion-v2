#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."

ALPHAS=(0.0 0.25 0.5 0.75 1.0)
ALIGNER_VERSIONS=("v1")
ALIGNER_DATASETS=("coco")
ALIGNER_LOSSES=("infonce")

FUSION_TOKEN_TYPES=("all")
FUSION_TYPES=("cross_attention")
USE_PNO_OPTIONS=("false")

PROMPT="a photo of a cat"
REF_IMG="data/van_gogh_starry_night.jpg"
CONFIG="configs/stable-diffusion/v2-inference-v.yaml"
CKPT="/scratch-shared/holy-triangle/weights/stable-diffusion-2-1/v2-1_768-ema-pruned.ckpt"
JOB_SCRIPT="${SCRIPT_DIR}/test_merge_single_run.sh"

mkdir -p "${SCRIPT_DIR}/../logs"

echo "Starting experiment with:"
echo "  - Alpha values: ${ALPHAS[*]}"
echo "  - Fusion token types: ${FUSION_TOKEN_TYPES[*]}"
echo "  - Fusion methods: ${FUSION_TYPES[*]}"
echo "  - PNO modes: ${USE_PNO_OPTIONS[*]}"
echo "  - Total jobs: $((${#ALPHAS[@]} * ${#FUSION_TOKEN_TYPES[@]} * ${#FUSION_TYPES[@]} * ${#USE_PNO_OPTIONS[@]} * ${#ALIGNER_VERSIONS[@]} * ${#ALIGNER_DATASETS[@]} * ${#ALIGNER_LOSSES[@]}))"
echo ""

for USE_PNO in "${USE_PNO_OPTIONS[@]}"; do
  for ALPHA in "${ALPHAS[@]}"; do
    for FUSION_TOKEN_TYPE in "${FUSION_TOKEN_TYPES[@]}"; do
      for FUSION_TYPE in "${FUSION_TYPES[@]}"; do
        for ALIGNER_VERSION in "${ALIGNER_VERSIONS[@]}"; do
          for ALIGNER_DATASET in "${ALIGNER_DATASETS[@]}"; do
            for ALIGNER_LOSS in "${ALIGNER_LOSSES[@]}"; do
              echo "Submitting job with alpha=${ALPHA}, token_type=${FUSION_TOKEN_TYPE}, fusion_type=${FUSION_TYPE}, PNO=${USE_PNO}, version=${ALIGNER_VERSION}, dataset=${ALIGNER_DATASET}, loss=${ALIGNER_LOSS}"
              sbatch --export=ALL,ALPHA=$ALPHA,PROMPT="$PROMPT",REF_IMG="$REF_IMG",CONFIG="$CONFIG",CKPT="$CKPT",ROOT_DIR="$ROOT_DIR",FUSION_TOKEN_TYPE="$FUSION_TOKEN_TYPE",FUSION_TYPE="$FUSION_TYPE",USE_PNO="$USE_PNO",ALIGNER_VERSION="$ALIGNER_VERSION",ALIGNER_DATASET="$ALIGNER_DATASET",ALIGNER_LOSS="$ALIGNER_LOSS" "$JOB_SCRIPT"
            done
          done
        done
      done
    done
  done
done
