#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/home/%u/thijmen/stable-diffusion-v2/outputs/jobs/run_alpha_${ALPHA}_%A.out
#SBATCH --time=01:00:00
#SBATCH --mem=40G

echo "Running job for alpha=${ALPHA} in $ROOT_DIR"

cd "$ROOT_DIR"
source .venv/bin/activate

uv run python scripts/txt2img.py \
  --prompt "$PROMPT" \
  --ckpt "$CKPT" \
  --config "$CONFIG" \
  --H 768 --W 768 \
  --ref_img "$REF_IMG" \
  --ref_blend_weight "$ALPHA" \
  --aligner_model_path "$ALIGNER_MODEL"
