#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-full}" # smoke | full
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

PROJECT="${WANDB_PROJECT:-cs336-opsd}"
GROUP="${WANDB_GROUP:-opsd_batch_${TIMESTAMP}}"
SAVE_ROOT="${SAVE_ROOT:-$ROOT_DIR/data/opsd_ckpts/${GROUP}}"
PYTHON_CMD="${PYTHON_CMD:-uv run python}"

MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/data/a5-alignment/models/Qwen2.5-Math-1.5B}"
PROMPT_PATH="${PROMPT_PATH:-$ROOT_DIR/cs336_alignment/prompts/r1_zero.prompt}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$ROOT_DIR/data/gsm8k/train.jsonl}"
TEST_DATA_PATH="${TEST_DATA_PATH:-$ROOT_DIR/data/gsm8k/test.jsonl}"

SEED="${SEED:-69}"
DEVICE_TRAIN="${DEVICE_TRAIN:-cuda:0}"
DEVICE_VLLM="${DEVICE_VLLM:-cuda:1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.3}"

PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-256}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
EPOCHS_PER_STEP="${EPOCHS_PER_STEP:-2}"
TARGET_UPDATES="${TARGET_UPDATES:-300}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
KL_COEF="${KL_COEF:-1.0}"
ROLLOUT_MAX_TOKENS="${ROLLOUT_MAX_TOKENS:-512}"
EVAL_STEPS="${EVAL_STEPS:-16}"

UPDATES_PER_STEP=$(( (PROMPT_BATCH_SIZE / (MICRO_BATCH_SIZE * GRAD_ACCUM)) * EPOCHS_PER_STEP ))
if (( UPDATES_PER_STEP <= 0 )); then
  UPDATES_PER_STEP=1
fi

if [[ "$MODE" == "smoke" ]]; then
  N_OPSD_STEPS="${N_OPSD_STEPS:-2}"
else
  DEFAULT_STEPS=$(( (TARGET_UPDATES + UPDATES_PER_STEP - 1) / UPDATES_PER_STEP ))
  N_OPSD_STEPS="${N_OPSD_STEPS:-$DEFAULT_STEPS}"
fi

COMMON_ARGS=(
  --project "$PROJECT"
  --model_path "$MODEL_PATH"
  --prompt_path "$PROMPT_PATH"
  --train_data_path "$TRAIN_DATA_PATH"
  --test_data_path "$TEST_DATA_PATH"
  --seed "$SEED"
  --device_train "$DEVICE_TRAIN"
  --device_vllm "$DEVICE_VLLM"
  --gpu_memory_utilization "$GPU_MEM_UTIL"
  --n_opsd_steps "$N_OPSD_STEPS"
  --prompt_batch_size "$PROMPT_BATCH_SIZE"
  --epochs_per_step "$EPOCHS_PER_STEP"
  --micro_batch_size "$MICRO_BATCH_SIZE"
  --gradient_accumulation_steps "$GRAD_ACCUM"
  --learning_rate "$LEARNING_RATE"
  --kl_coef "$KL_COEF"
  --rollout_max_tokens "$ROLLOUT_MAX_TOKENS"
  --eval_steps "$EVAL_STEPS"
)

run_case() {
  local run_name="$1"
  local use_gt="$2"
  local tags="$3"
  shift 3
  local extra_args=("$@")
  local teacher_flag="--teacher_use_ground_truth"
  if [[ "$use_gt" != "true" ]]; then
    teacher_flag="--no-teacher_use_ground_truth"
  fi

  local run_dir="${SAVE_ROOT}/${run_name}"
  echo "============================================================"
  echo "Running OPSD: $run_name"
  echo "  teacher_use_ground_truth=$use_gt"
  echo "  n_opsd_steps=$N_OPSD_STEPS"
  echo "  updates_per_step=$UPDATES_PER_STEP"
  echo "  target_updates=$TARGET_UPDATES"
  echo "  group=$GROUP"
  echo "============================================================"

  $PYTHON_CMD cs336_alignment/train_opsd.py \
    "${COMMON_ARGS[@]}" \
    --run_name "$run_name" \
    --wandb_group "$GROUP" \
    --wandb_job_type "opsd_ablation" \
    --wandb_tags "$tags" \
    --wandb_notes "group=${GROUP};mode=${MODE};gt=${use_gt};updates_per_step=${UPDATES_PER_STEP};target_updates=${TARGET_UPDATES}" \
    --output_path "$run_dir" \
    "$teacher_flag" \
    "${extra_args[@]}"
}

# 1) OPSD with privileged answer hint (paper-style)
run_case "${GROUP}_opsd01_gt_on" "true" "opsd,gt_on,ablation"

# 2) Self-distill control: same pipeline but no privileged answer hint
run_case "${GROUP}_opsd02_gt_off" "false" "opsd,gt_off,ablation"

echo
echo "OPSD batch finished."
echo "W&B project: $PROJECT"
echo "W&B group:   $GROUP"
echo "Save root:   $SAVE_ROOT"
