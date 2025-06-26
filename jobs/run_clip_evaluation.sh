#!/bin/bash

echo "Launching parallel CLIP evaluations..."

# Path to project root
ROOT_DIR="$(pwd)"
JOBS_DIR="${ROOT_DIR}/jobs"

# Create evaluation results directory
EVAL_DIR="${ROOT_DIR}/outputs/evaluations"
mkdir -p "$EVAL_DIR"

# Define evaluation methods - you can easily modify this list
METHODS=(
  "alpha_blend:0.0"
  "alpha_blend:0.3"
  "concat:none"
)

# Path to the single job script
JOB_SCRIPT="${JOBS_DIR}/run_single_clip_eval.sh"

echo "=== Submitting Parallel CLIP Evaluation Jobs ==="
echo "Results will be saved to: $EVAL_DIR"
echo "Using CLIP model: ViT-B/32"
echo "Methods to evaluate: ${#METHODS[@]}"
for method in "${METHODS[@]}"; do
  echo "  - $method"
done
echo "Total jobs to submit: ${#METHODS[@]}"
echo ""

# Clean up any existing shared summary file to start fresh
SHARED_RESULTS_FILE="${EVAL_DIR}/clip_summary.json"
if [ -f "$SHARED_RESULTS_FILE" ]; then
  echo "Removing existing shared summary file: $SHARED_RESULTS_FILE"
  rm "$SHARED_RESULTS_FILE"
fi

# Submit jobs for each method
JOB_IDS=()
JOB_NAMES=()
OUTPUT_FILES=()

job_counter=1
for METHOD in "${METHODS[@]}"; do
  FUSION_TYPE=$(echo "$METHOD" | cut -d':' -f1)
  ALPHA=$(echo "$METHOD" | cut -d':' -f2)
  
  # Create descriptive names
  if [ "$ALPHA" = "none" ]; then
    JOB_NAME="clip_${FUSION_TYPE}"
    OUTPUT_FILE="${EVAL_DIR}/clip_${FUSION_TYPE}.json"
  else
    JOB_NAME="clip_${FUSION_TYPE}_${ALPHA}"
    OUTPUT_FILE="${EVAL_DIR}/clip_${FUSION_TYPE}_${ALPHA}.json"
  fi
  
  echo "${job_counter}. Submitting ${FUSION_TYPE} (α=${ALPHA}) evaluation..."
  JOB_ID=$(sbatch \
    --job-name="$JOB_NAME" \
    --output="outputs/jobs/${JOB_NAME}_%A.out" \
    --export=ALL,FUSION_TYPE="$FUSION_TYPE",ALPHA="$ALPHA",OUTPUT_FILE="$OUTPUT_FILE",ROOT_DIR="$ROOT_DIR" \
    "$JOB_SCRIPT" | awk '{print $4}')
  
  echo "  Job ID: $JOB_ID"
  
  JOB_IDS+=("$JOB_ID")
  JOB_NAMES+=("$JOB_NAME")
  OUTPUT_FILES+=("$OUTPUT_FILE")
  
  job_counter=$((job_counter + 1))
done

echo ""
echo "=== All CLIP Evaluation Jobs Submitted ==="
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo ""
echo "Check specific job logs:"
for i in "${!JOB_NAMES[@]}"; do
  echo "  tail -f outputs/jobs/${JOB_NAMES[$i]}_${JOB_IDS[$i]}.out"
done
echo ""
echo "Individual results will be saved to:"
for output_file in "${OUTPUT_FILES[@]}"; do
  echo "  - $output_file"
done
echo ""
echo "Shared summary will be saved to:"
echo "  - $SHARED_RESULTS_FILE"
echo ""
echo "To check when all jobs are done:"
echo "  squeue -j $(IFS=,; echo "${JOB_IDS[*]}")" 