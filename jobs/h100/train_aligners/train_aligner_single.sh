python scripts/vcf/train_img2text_aligner.py \
  --datasets coco \
  --loss cosine \
  --batch_size 256 \
  --epochs 10 \
  --lr 1e-4 \
  --device cuda \
  --model_path weights/img2text_aligner/coco_cosine/model.pth \
  --save_every 2