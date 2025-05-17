# 💾 Download Model Weights

⚠️ **Note:** Execute the following **on a compute node**, not the login node, to prevent issues with large downloads.

Start an interactive GPU session:

```bash
srun --partition=gpu_h100 --gpus=1 --ntasks=1 --cpus-per-task=9 --time=00:30:00 --pty bash -i
```

Then:

1. Install Git LFS (locally, without sudo):

```bash
chmod +x ./install_git_lfs.sh
./install_git_lfs.sh
```

2. Set up Git LFS and download model weights:

```bash
git lfs install
git clone https://huggingface.co/stabilityai/stable-diffusion-2-1 /scratch-shared/holy-triangle/weights/stable-diffusion-2-1
```

---

# 🚀 Using UV

## 📦 Set Up the Environment

First, install [`uv`](https://github.com/astral-sh/uv):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, initialize the project environment:

```bash
uv venv
source .venv/bin/activate
uv sync                 # Install from pyproject.toml and uv.lock (if available)
uv pip install -e .     # Install your project in editable mode
```

---

## 🖼️ Generate Images with Stable Diffusion 2.1

Load required modules and run the model:

```bash
module load 2022
module load CUDA/11.7.0
```

```bash
uv run python scripts/txt2img.py \
  --prompt "a professional photograph of an astronaut riding a horse" \
  --ckpt ./model_checkpoint.ckpt \
  --config configs/stable-diffusion/v2-inference-v.yaml \
  --H 768 --W 768 \
  --ref_img "data/picasso_style.jpg" \
  --ref_blend_weight 0 \
  --aligner_model_path model_best_cosine_flickr30k.pth
```

---

# 🐍 Using Conda

1. Log in to a compute node (e.g., A100):

```bash
module load 2023
module spider CUDA/12.6.0
module load Anaconda3/2023.07-2  # or try Anaconda3/2022.*
```

2. Create and activate the environment:

```bash
conda env create -f environment.yaml
source activate ldmv2
```

3. Run image generation:

```bash
uv run python scripts/txt2img.py \
  --prompt "a professional photograph of an astronaut riding a horse" \
  --ckpt /scratch-shared/holy-triangle/weights/stable-diffusion-2-1/v2-1_768-ema-pruned.ckpt \
  --config configs/stable-diffusion/v2-inference-v.yaml \
  --H 768 --W 768 \
  --ref_img "data/picasso_style.jpg"
```


export HF_DATASETS_CACHE="/scratch-shared/holy-triangle/huggingface_datasets2"
export TRANSFORMERS_CACHE="/scratch-shared/holy-triangle/huggingface_models2"
export WANDB_API_KEY="a073531c9973b3a17a47501f4c98affd7d2f3c8c"

uv run python scripts/txt2img.py \
  --prompt "a photo of a cat" \
  --ckpt /scratch-shared/holy-triangle/weights/stable-diffusion-2-1/v2-1_768-ema-pruned.ckpt \
  --config configs/stable-diffusion/v2-inference-v.yaml \
  --H 768 --W 768 \
  --ref_img "data/picasso_style.jpg"
  --ref_blend_weight 0 \
  --aligner_model_path model_best_cosine_flickr30k.pth

uv run python scripts/vcf/train_img2text_aligner.py \
  --datasets 'flickr30k' \
  --loss cosine \
  --batch_size 256 \
  --epochs 10 \
  --lr 1e-4 \
  --device cuda \
  --wandb_project text-image-aligner \
  --model_path weights/img2text_aligner/coco_cosine/model.pth \
  --save_every 2


uv run python scripts/txt2img.py \
  --prompt "a photo of a cat" \
  --ckpt "model_checkpoint.ckpt" \
  --config "configs/stable-diffusion/v2-inference-v.yaml" \
  --H 768 --W 768 \
  --ref_img "data/cat.jpg" \
  --ref_blend_weight 0 \
  --aligner_model_path "/scratch-shared/holy-triangle/weights/img2text_aligner_fixed/flickr30k_cosine/model_best.pth"
