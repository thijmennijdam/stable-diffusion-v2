#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --output=outputs/jobs/run_alpha_${ALPHA}_%A.out
#SBATCH --time=01:00:00
#SBATCH --mem=40G

echo "Running job for alpha=${ALPHA} in $ROOT_DIR"

cd "$ROOT_DIR"

# When using UV on a100
# source .venv/bin/activate

module load 2023
module load Anaconda3/2023.07-2
module load CUDA/12.1.1

# conda env create -f environment.yaml
source activate ldmv2
# conda install pytorch torchvision=0.18.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Build the base command using individual aligner parameters (matching main_results.sh)
CMD="uv run python scripts/txt2img.py \
  --prompt \"$PROMPT\" \
  --ckpt \"$CKPT\" \
  --config \"$CONFIG\" \
  --H 768 --W 768 \
  --ref_img \"$REF_IMG\" \
  --ref_blend_weight \"$ALPHA\" \
  --aligner_version v1 \
  --aligner_dataset coco \
  --aligner_loss \"$ALIGNER_LOSS\" \
  --fusion_token_type \"$FUSION_TOKEN_TYPE\" \
  --fusion_type \"$FUSION_TYPE\" \
  --calculate_clip_score"


echo "Running: $CMD"
eval $CMD