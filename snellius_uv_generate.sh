#!/usr/bin/env bash
set -euo pipefail

# Usage: ./uv_generate.sh "your prompt here"
PROMPT="${1:-}"
if [[ -z "$PROMPT" ]]; then
  echo "Usage: $0 \"your text prompt here\""
  exit 1
fi

# Load GPU modules
module purge
module load 2022
module load CUDA/11.8.0
# module load Python/3.10.4-GCCcore-11.3.0


# Activate uv venv
# shellcheck disable=SC1091
source .venv/bin/activate

# Run txt2img via uv
uv run python scripts/txt2img.py \
  --prompt "$PROMPT" \
  --ckpt weights/stable-diffusion-2-1/v2-1_768-ema-pruned.ckpt \
  --config configs/stable-diffusion/v2-inference-v.yaml \
  --H 768 --W 768
