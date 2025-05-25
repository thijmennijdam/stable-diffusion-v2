#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/home/scur0548/thijmen/stable-diffusion-v2/outputs/jobs/test_aligners/run_alpha_%A.out
#SBATCH --time=00:10:00

echo "Running job for alpha=${ALPHA} in $ROOT_DIR"

cd "$ROOT_DIR"


module load 2023
module load Anaconda3/2023.07-2
module load CUDA/12.1.1

# conda env create -f environment.yaml
source activate ldmv2
# conda install pytorch torchvision=0.18.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

python scripts/txt2img.py \
  --prompt "$PROMPT" \
  --ckpt "$CKPT" \
  --config "$CONFIG" \
  --ref_img "$REF_IMG" \
  --ref_blend_weight "$ALPHA" \
  --aligner_version "$ALIGNER_VERSION" \
  --aligner_dataset "$ALIGNER_DATASET" \
  --aligner_loss "$ALIGNER_LOSS" \
  --aligner_batch_size "$ALIGNER_BATCH_SIZE" \
  --aligner_dropout "$ALIGNER_DROPOUT" \
  --aligner_exclude_cls "$ALIGNER_CLS"

# uv run python scripts/txt2img.py \
#   --prompt "a photo of a cat" \
#   --ckpt "model_checkpoint.ckpt" \
#   --config "configs/stable-diffusion/v2-inference-v.yaml" \
#   --H 768 --W 768 \
#   --ref_img "data/cat.jpg" \
#   --ref_blend_weight "0.5" \
#   --aligner_version "v1" \
#   --aligner_dataset "coco" \
#   --aligner_loss "mse" \
#   --aligner_batch_size "64" \
#   --aligner_dropout "0.1" \
#   --aligner_exclude_cls "false"

# python scripts/vcf/train_img2text_alignerv2.py \
