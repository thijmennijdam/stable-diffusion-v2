# alpha_blend
uv run python scripts/txt2img.py \
  --prompt "a photo of a cat" \
  --ckpt ./weights/v2-1_768-ema-pruned.ckpt \
  --config configs/stable-diffusion/v2-inference-v.yaml \
  --H 768 --W 768 \
  --ref_img data/cat.jpg \
  --ref_blend_weight 0.3 \
  --aligner_version v1 \
  --aligner_dataset coco \
  --aligner_loss combined \
  --fusion_token_type all \
  --fusion_type alpha_blend \
  --calculate_clip_score

# concat
uv run python scripts/txt2img.py \
  --prompt "a photo of a cat" \
  --ckpt ./weights/v2-1_768-ema-pruned.ckpt \
  --config configs/stable-diffusion/v2-inference-v.yaml \
  --H 768 --W 768 \
  --ref_img data/cat.jpg \
  --ref_blend_weight 0.3 \
  --aligner_version v1 \
  --aligner_dataset coco \
  --aligner_loss combined \
  --fusion_token_type all \
  --fusion_type concat \
  --calculate_clip_score

# cross_attention
uv run python scripts/txt2img.py \
  --prompt "a photo of a cat" \
  --ckpt ./weights/v2-1_768-ema-pruned.ckpt \
  --config configs/stable-diffusion/v2-inference-v.yaml \
  --H 768 --W 768 \
  --ref_img data/cat.jpg \
  --ref_blend_weight 0.3 \
  --aligner_version v1 \
  --aligner_dataset coco \
  --aligner_loss combined \
  --fusion_token_type all \
  --fusion_type cross_attention \
  --calculate_clip_score