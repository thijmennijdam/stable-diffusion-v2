#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=20G

echo "Starting CLIP evaluation for: fusion_type=${FUSION_TYPE}, alpha=${ALPHA}"
echo "Job ID: $SLURM_JOB_ID"

# Path to project root
ROOT_DIR="$(pwd)"
cd "$ROOT_DIR"

# When using UV on a100 - fix VIRTUAL_ENV warning
source .venv/bin/activate
unset VIRTUAL_ENV

# Create evaluation results directory
EVAL_DIR="${ROOT_DIR}/outputs/evaluations"
mkdir -p "$EVAL_DIR"

# Base path for experiments
BASE_PATH="${ROOT_DIR}/outputs/txt2img-samples"

echo "=== Running CLIP Evaluation ==="
echo "Base path: $BASE_PATH"
echo "Fusion type: $FUSION_TYPE"
echo "Alpha: $ALPHA"
echo "Output file: $OUTPUT_FILE"
echo ""

# Build the command based on whether alpha is specified
if [ "$ALPHA" = "none" ]; then
    # For concat fusion (no alpha parameter)
    CMD="uv run python scripts/quantitative_metrics/compute_clip_score.py \
      --base_path \"$BASE_PATH\" \
      --fusion_type \"$FUSION_TYPE\" \
      --output_file \"$OUTPUT_FILE\""
else
    # For alpha_blend fusion (with alpha parameter)
    CMD="uv run python scripts/quantitative_metrics/compute_clip_score.py \
      --base_path \"$BASE_PATH\" \
      --fusion_type \"$FUSION_TYPE\" \
      --alpha \"$ALPHA\" \
      --output_file \"$OUTPUT_FILE\""
fi

echo "Running: $CMD"
eval $CMD

# Check if evaluation was successful
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "=== Evaluation Complete ==="
    echo "Results saved to: $OUTPUT_FILE"
    
    # Print quick summary
    SUMMARY=$(python -c "
import json
try:
    with open('$OUTPUT_FILE') as f:
        data = json.load(f)
    summary = data['summary']
    print(f\"Processed {summary['total_samples']} samples from {summary['total_folders']} folders\")
    if summary['overall_avg_combined_clip'] is not None:
        print(f\"Average Text-to-Image CLIP: {summary['overall_avg_text_to_image_clip']:.4f} (±{summary['overall_std_text_to_image_clip']:.4f})\")
        print(f\"Average Image-to-Image CLIP: {summary['overall_avg_image_to_image_clip']:.4f} (±{summary['overall_std_image_to_image_clip']:.4f})\")
        print(f\"Average Combined CLIP: {summary['overall_avg_combined_clip']:.4f} (±{summary['overall_std_combined_clip']:.4f})\")
        print(f\"Combined CLIP Range: {summary['overall_min_combined_clip']:.4f} - {summary['overall_max_combined_clip']:.4f}\")
    else:
        print('No valid results found')
except Exception as e:
    print(f'Error reading results: {e}')
")
    
    echo "$SUMMARY"
    
    # Append summary to shared results file
    SHARED_RESULTS_FILE="${EVAL_DIR}/clip_summary.json"
    echo "Appending summary to shared results file: $SHARED_RESULTS_FILE"
    
    python -c "
import json
import os
from datetime import datetime

# Load individual results
with open('$OUTPUT_FILE') as f:
    data = json.load(f)

# Create summary entry for shared file
summary_entry = {
    'method': '$FUSION_TYPE' + ('_alpha' + '$ALPHA' if '$ALPHA' != 'none' else ''),
    'fusion_type': '$FUSION_TYPE',
    'alpha': '$ALPHA' if '$ALPHA' != 'none' else None,
    'timestamp': datetime.now().isoformat(),
    'job_id': os.getenv('SLURM_JOB_ID', 'unknown'),
    'total_samples': data['summary']['total_samples'],
    'total_folders': data['summary']['total_folders'],
    'avg_text_to_image_clip': data['summary']['overall_avg_text_to_image_clip'],
    'avg_image_to_image_clip': data['summary']['overall_avg_image_to_image_clip'],
    'avg_combined_clip': data['summary']['overall_avg_combined_clip'],
    'std_text_to_image_clip': data['summary']['overall_std_text_to_image_clip'],
    'std_image_to_image_clip': data['summary']['overall_std_image_to_image_clip'],
    'std_combined_clip': data['summary']['overall_std_combined_clip'],
    'min_text_to_image_clip': data['summary']['overall_min_text_to_image_clip'],
    'max_text_to_image_clip': data['summary']['overall_max_text_to_image_clip'],
    'min_image_to_image_clip': data['summary']['overall_min_image_to_image_clip'],
    'max_image_to_image_clip': data['summary']['overall_max_image_to_image_clip'],
    'min_combined_clip': data['summary']['overall_min_combined_clip'],
    'max_combined_clip': data['summary']['overall_max_combined_clip'],
    'detailed_file': '$OUTPUT_FILE'
}

# Load or create shared results file
shared_file = '$SHARED_RESULTS_FILE'
if os.path.exists(shared_file):
    with open(shared_file) as f:
        shared_data = json.load(f)
else:
    shared_data = {
        'evaluation_info': {
            'model': 'ViT-B/32',
            'base_path': '$BASE_PATH',
            'created': datetime.now().isoformat()
        },
        'results': []
    }

# Add new result
shared_data['results'].append(summary_entry)

# Sort results by method name for consistency
shared_data['results'].sort(key=lambda x: x['method'])

# Save shared results
with open(shared_file, 'w') as f:
    json.dump(shared_data, f, indent=2)

print(f'Summary appended to {shared_file}')
"
else
    echo "ERROR: Evaluation failed - output file not created"
    exit 1
fi

echo ""
echo "CLIP evaluation job completed successfully!" 