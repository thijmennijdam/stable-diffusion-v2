#!/bin/bash

echo "✅ Job started at: $(date)"
echo "Alpha: ${ALPHA}"
echo "Prompt: ${PROMPT}"
echo "Ref Image: ${REF_IMG}"
echo "Aligner Model: ${ALIGNER_MODEL}"
echo "Config: ${CONFIG}"
echo "Checkpoint: ${CKPT}"
echo "Root Dir: ${ROOT_DIR}"

module purge
module load 2023
module load Anaconda3/2023.07-2
module load CUDA/12.1.1

source activate ldmv2

cd "$ROOT_DIR" || exit 1

python scripts/txt2img.py \
  --prompt "$PROMPT" \
  --ckpt "$CKPT" \
  --config "$CONFIG" \
  --H 768 --W 768 \
  --ref_img "$REF_IMG" \
  --ref_blend_weight "$ALPHA" \
  --aligner_model_path "$ALIGNER_MODEL"
echo "✅ Job completed at: $(date)"