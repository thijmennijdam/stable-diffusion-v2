#!/bin/bash

echo "✅ Job started at: $(date)"
echo "Alpha: ${ALPHA}"
echo "Prompt: ${PROMPT}"
echo "Ref Image: ${REF_IMG}"

module load 2022
module load CUDA/11.7.0

cd "$ROOT_DIR" || exit 1

# source activate ldmv2
source .venv/bin/activate

cd "$ROOT_DIR"

# 🚀 PNO inference
python scripts/txt2img.py \
  --prompt "$PROMPT" \
  --ckpt "$CKPT" \
  --config "$CONFIG" \
  --H 512 --W 512 \
  --ref_blend_weight "$ALPHA" \
  --n_samples 1 \
  --n_iter 1 \
  --use_pno_trajectory \
  --ref_img "$REF_IMG" \
  --pno_steps 50 \
  --lr_prompt 1e-2 \
  --lr_noise 1e-2 \
  --pno_noise_reg_gamma 0.1 \
  --pno_clip_grad_norm 1.0 \
  --steps 50
