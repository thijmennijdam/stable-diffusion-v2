#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/home/%u/FOMO/stable-diffusion-v2/outputs/jobs/run_alpha_${ALPHA}_%A.out
#SBATCH --time=01:00:00

echo "✅ Job started at: $(date)"
module purge
module load 2023
module load Anaconda3/2023.07-2
module load CUDA/12.1.1

source activate ldmv2

echo "Running job for alpha=${ALPHA} in $ROOT_DIR"

cd "$ROOT_DIR"


python scripts/txt2img.py \
  --prompt "$PROMPT" \
  --ckpt "$CKPT" \
  --config "$CONFIG" \
  --H 768 --W 768 \
  --ref_img "$REF_IMG" \
  --ref_blend_weight "$ALPHA" \
  --aligner_model_path "$ALIGNER_MODEL"
