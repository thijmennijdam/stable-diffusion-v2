#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --output=outputs/jobs/run_alpha_${ALPHA}_pno_${USE_PNO}_fusion_${FUSION_TYPE}_%A.out
#SBATCH --time=00:10:00

echo "✅ Job started at: $(date)"
echo "Alpha: ${ALPHA}"
echo "Prompt: ${PROMPT}"
echo "Ref Image: ${REF_IMG}"
echo "Using PNO: ${USE_PNO}"

cd "$ROOT_DIR" || exit 1

# Module loading
module load 2022
module load CUDA/11.7.0

source .venv/bin/activate

uv pip install -e .

# Determine which command to run
if [ "$USE_PNO" = "true" ]; then
  echo "🔁 Running with PNO inference"
  CMD="python scripts/txt2img.py \
    --prompt \"$PROMPT\" \
    --ckpt \"$CKPT\" \
    --config \"$CONFIG\" \
    --H 512 --W 512 \
    --ref_img \"$REF_IMG\" \
    --ref_blend_weight \"$ALPHA\" \
    --aligner_version \"$ALIGNER_VERSION\" \
    --aligner_dataset \"$ALIGNER_DATASET\" \
    --aligner_loss \"$ALIGNER_LOSS\" \
    --fusion_token_type \"$FUSION_TOKEN_TYPE\" \
    --fusion_type \"$FUSION_TYPE\" \
    --n_samples 1 \
    --n_iter 1 \
    --use_pno_trajectory \
    --pno_steps 50 \
    --lr_prompt 1e-2 \
    --lr_noise 1e-2 \
    --pno_noise_reg_gamma 0.1 \
    --pno_clip_grad_norm 1.0"
else
  echo "📸 Running standard fusion inference"
  CMD="python scripts/txt2img.py \
    --prompt \"$PROMPT\" \
    --ckpt \"$CKPT\" \
    --config \"$CONFIG\" \
    --H 768 --W 768 \
    --ref_img \"$REF_IMG\" \
    --ref_blend_weight \"$ALPHA\" \
    --aligner_version \"$ALIGNER_VERSION\" \
    --aligner_dataset \"$ALIGNER_DATASET\" \
    --aligner_loss \"$ALIGNER_LOSS\" \
    --fusion_token_type \"$FUSION_TOKEN_TYPE\" \
    --fusion_type \"$FUSION_TYPE\""
fi

# Run the command
echo "Running: $CMD"
eval $CMD

echo "✅ Job finished at: $(date)"
