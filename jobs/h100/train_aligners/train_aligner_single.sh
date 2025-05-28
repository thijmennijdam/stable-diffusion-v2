uv run python scripts/vcf/train_img2text_aligner.py \
  --datasets coco \
  --loss infonce \
  --batch_size 64 \
  --epochs 10 \
  --lr 1e-4 \
  --device cuda \
  --save_every 2 \
  --version v1