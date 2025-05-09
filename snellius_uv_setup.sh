#!/usr/bin/env bash
set -euo pipefail

# If not yet on a GPU node, re-submit this script under srun
if [[ -z "${SLURM_JOB_ID-}" ]]; then
  exec srun \
    --partition=gpu_a100 \
    --gpus=1 \
    --ntasks=1 \
    --cpus-per-task=9 \
    --time=00:20:00 \
    --pty bash "$0"
fi

echo ">>> On compute node: $(hostname)"

# remember where we started
ORIG_DIR="$(pwd)"

# 1) Install Git LFS locally
echo ">>> Installing Git LFS..."
chmod +x ./install_git_lfs.sh
./install_git_lfs.sh

# 2) Setup Git LFS and clone weights
echo ">>> Cloning Stable-Diffusion 2.1 weights..."
git lfs install
WEIGHTS_DIR="weights/stable-diffusion-2-1"
if [[ ! -d "$WEIGHTS_DIR" ]]; then
  git clone https://huggingface.co/stabilityai/stable-diffusion-2-1 "$WEIGHTS_DIR"
else
  echo "Weights already present, skipping."
fi

# 3) Install uv and bootstrap your project
echo ">>> Installing uv…"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.uv/bin:$PATH"

echo ">>> Creating venv & installing dependencies…"
uv venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv sync
uv pip install -e .

echo "✅ Setup complete. Now use ./uv_generate.sh to launch jobs."
