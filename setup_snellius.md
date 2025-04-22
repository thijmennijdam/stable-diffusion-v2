# 💾 Download Model Weights

⚠️ **Note:** Execute the following **on a compute node**, not the login node, to prevent issues with large downloads.

Start an interactive GPU session:

```bash
srun --partition=gpu_a100 --gpus=1 --ntasks=1 --cpus-per-task=9 --time=00:20:00 --pty bash -i
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
git clone https://huggingface.co/stabilityai/stable-diffusion-2-1 ./weights/stable-diffusion-2-1
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
module load CUDA/11.3.0
```

```bash
uv run python scripts/txt2img.py \
  --prompt "a professional photograph of an astronaut riding a horse" \
  --ckpt weights/stable-diffusion-2-1/v2-1_768-ema-pruned.ckpt \
  --config configs/stable-diffusion/v2-inference-v.yaml \
  --H 768 --W 768
```

---

# 🐍 Using Conda

1. Log in to a compute node (e.g., A100):

```bash
module load 2022
module load CUDA/11.3.0
module load Anaconda3/2023.07-2  # or try Anaconda3/2022.*
```

2. Create and activate the environment:

```bash
conda env create -f environment.yaml
source activate ldmv2
```

3. Run image generation:

```bash
python scripts/txt2img.py \
  --prompt "a professional photograph of an astronaut riding a horse" \
  --ckpt weights/stable-diffusion-2-1/v2-1_768-ema-pruned.ckpt \
  --config configs/stable-diffusion/v2-inference-v.yaml \
  --H 768 --W 768
```