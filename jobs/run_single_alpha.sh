#!/bin/bash

echo "✅ Job started at: $(date)"
echo "Alpha: ${ALPHA}"
echo "Prompt: ${PROMPT}"
echo "Ref Image: ${REF_IMG}"
echo "Aligner Model: ${ALIGNER_MODEL}"
echo "Config: ${CONFIG}"
echo "Checkpoint: ${CKPT}"
echo "Root Dir: ${ROOT_DIR}"

module load 2022
module load CUDA/11.7.0

cd "$ROOT_DIR" || exit 1

# source activate ldmv2
source .venv/bin/activate

uv run python scripts/txt2img.py \
  --prompt "$PROMPT" \
  --ckpt "$CKPT" \
  --config "$CONFIG" \
  --H 768 --W 768 \
  --ref_img "$REF_IMG" \
  --ref_blend_weight "$ALPHA" \
  --aligner_model_path "$ALIGNER_MODEL"
echo "✅ Job completed at: $(date)"

# uv run python scripts/txt2img.py \
#   --prompt "a photo of a cat" \
#   --ckpt "/scratch-shared/holy-triangle/weights/stable-diffusion-2-1/v2-1_768-ema-pruned.ckpt" \
#   --config "/home/scur2690/stable-diffusion-v2/configs/stable-diffusion/v2-inference-v.yaml" \
#   --H 768 --W 768 \
#   --ref_img "/home/scur2690/stable-diffusion-v2/data/van_gogh_starry_night.jpg" \
#   --ref_blend_weight 0.1 \
#   --aligner_model_path "/scratch-shared/holy-triangle/weights/img2text_aligner_fixed/flickr30k_infonce/model_best.pth"
