#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-smoke}" # smoke | full | token_fill | kl_ablation | dapo_ablation | gbpo_ablation
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

PROJECT="${WANDB_PROJECT:-cs336-anypo2}"
GROUP="${WANDB_GROUP:-anypo_batch_${TIMESTAMP}}"
SAVE_ROOT="${SAVE_ROOT:-$ROOT_DIR/data/anypo_ckpts/${GROUP}}"
PYTHON_CMD="${PYTHON_CMD:-uv run python}"

if [[ "$MODE" == "smoke" ]]; then
  N_GRPO_STEPS="${N_GRPO_STEPS:-4}"
  EVAL_FREQ="${EVAL_FREQ:-2}"
  EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-128}"
else
  N_GRPO_STEPS="${N_GRPO_STEPS:-300}"
  EVAL_FREQ="${EVAL_FREQ:-8}"
  EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-0}"
fi

SEED="${SEED:-69}"
DEVICE_TRAIN="${DEVICE_TRAIN:-cuda:0}"
DEVICE_VLLM="${DEVICE_VLLM:-cuda:1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
GROUP_SIZE="${GROUP_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.3}"

COMMON_ARGS=(
  --project "$PROJECT"
  --wandb_group "$GROUP"
  --wandb_job_type "batch_ablation"
  --seed "$SEED"
  --device_train "$DEVICE_TRAIN"
  --device_vllm "$DEVICE_VLLM"
  --gpu_memory_utilization "$GPU_MEM_UTIL"
  --n_grpo_steps "$N_GRPO_STEPS"
  --eval_freq "$EVAL_FREQ"
  --eval_num_samples "$EVAL_NUM_SAMPLES"
  --rollout_batch_size "$ROLLOUT_BATCH_SIZE"
  --train_batch_size "$TRAIN_BATCH_SIZE"
  --group_size "$GROUP_SIZE"
  --gradient_accumulation_steps "$GRAD_ACCUM"
  --save_best
)

run_case() {
  local run_name="$1"
  local loss_type="$2"
  local loss_aggregation="$3"
  local std_flag="$4"
  local tags="$5"
  shift 5
  local extra_args=("$@")

  echo "============================================================"
  echo "Running: $run_name"
  echo "  loss_type=$loss_type"
  echo "  loss_aggregation=$loss_aggregation"
  echo "  std_flag=$std_flag"
  echo "  group=$GROUP"
  echo "============================================================"
  local run_save_dir="${SAVE_ROOT}/${run_name}"

  $PYTHON_CMD cs336_alignment/train_anypo.py \
    "${COMMON_ARGS[@]}" \
    --run_name "$run_name" \
    --loss_type "$loss_type" \
    --loss_aggregation "$loss_aggregation" \
    "$std_flag" \
    --save_dir "$run_save_dir" \
    --wandb_tags "$tags" \
    --wandb_notes "batch=${GROUP};mode=${MODE};loss=${loss_type};agg=${loss_aggregation};std=${std_flag}" \
    "${extra_args[@]}"
}

run_full_or_smoke() {
  # 1) grpo_clip + std_off + seq
  run_case "${GROUP}_01_clip_seq_std0" "grpo_clip" "seq" "--no-use_std_normalization" \
    "batch,anypo,grpo_clip,seq,std0,ablation"

  # 2) grpo_no_clip + std_off + seq
  run_case "${GROUP}_02_noclip_seq_std0" "grpo_no_clip" "seq" "--no-use_std_normalization" \
    "batch,anypo,grpo_no_clip,seq,std0,ablation"

  # 3) reinforce_with_baseline + std_off + seq
  run_case "${GROUP}_03_reinforce_seq_std0" "reinforce_with_baseline" "seq" "--no-use_std_normalization" \
    "batch,anypo,reinforce_with_baseline,seq,std0,ablation"

  # 4) no_baseline + std_off + seq
  run_case "${GROUP}_04_nobase_seq_std0" "no_baseline" "seq" "--no-use_std_normalization" \
    "batch,anypo,no_baseline,seq,std0,ablation"

  # 5) grpo_clip + std_on + seq
  run_case "${GROUP}_05_clip_seq_std1" "grpo_clip" "seq" "--use_std_normalization" \
    "batch,anypo,grpo_clip,seq,std1,ablation"

  # 6) grpo_no_clip + std_on + seq
  run_case "${GROUP}_06_noclip_seq_std1" "grpo_no_clip" "seq" "--use_std_normalization" \
    "batch,anypo,grpo_no_clip,seq,std1,ablation"

  # 7) grpo_clip + std_off + token
  run_case "${GROUP}_07_clip_token_std0" "grpo_clip" "token" "--no-use_std_normalization" \
    "batch,anypo,grpo_clip,token,std0,ablation"

  # 8) grpo_no_clip + std_off + token
  run_case "${GROUP}_08_noclip_token_std0" "grpo_no_clip" "token" "--no-use_std_normalization" \
    "batch,anypo,grpo_no_clip,token,std0,ablation"
}

run_token_fill() {
  # A) reinforce_with_baseline + std_off + token
  run_case "${GROUP}_09_reinforce_token_std0" "reinforce_with_baseline" "token" "--no-use_std_normalization" \
    "batch,anypo,reinforce_with_baseline,token,std0,token_fill"

  # B) no_baseline + std_off + token
  run_case "${GROUP}_10_nobase_token_std0" "no_baseline" "token" "--no-use_std_normalization" \
    "batch,anypo,no_baseline,token,std0,token_fill"

  # C) grpo_clip + std_on + token
  run_case "${GROUP}_11_clip_token_std1" "grpo_clip" "token" "--use_std_normalization" \
    "batch,anypo,grpo_clip,token,std1,token_fill"

  # D) grpo_no_clip + std_on + token
  run_case "${GROUP}_12_noclip_token_std1" "grpo_no_clip" "token" "--use_std_normalization" \
    "batch,anypo,grpo_no_clip,token,std1,token_fill"
}

run_kl_ablation() {
  # Token aggregation: no KL vs +KL
  run_case "${GROUP}_kl00_clip_token_std0_no_kl" "grpo_clip" "token" "--no-use_std_normalization" \
    "batch,anypo,grpo_clip,token,std0,kl0,ablation" \
    --no-use_kl_penalty

  run_case "${GROUP}_kl01_clip_token_std0_with_kl" "grpo_clip" "token" "--no-use_std_normalization" \
    "batch,anypo,grpo_clip,token,std0,kl1,ablation" \
    --use_kl_penalty \
    --kl_beta "${KL_BETA:-0.02}" \
    --kl_estimator "${KL_ESTIMATOR:-k3}"

  # Seq aggregation: no KL vs +KL
  run_case "${GROUP}_kl00_clip_seq_std0_no_kl" "grpo_clip" "seq" "--no-use_std_normalization" \
    "batch,anypo,grpo_clip,seq,std0,kl0,ablation" \
    --no-use_kl_penalty

  run_case "${GROUP}_kl01_clip_seq_std0_with_kl" "grpo_clip" "seq" "--no-use_std_normalization" \
    "batch,anypo,grpo_clip,seq,std0,kl1,ablation" \
    --use_kl_penalty \
    --kl_beta "${KL_BETA:-0.02}" \
    --kl_estimator "${KL_ESTIMATOR:-k3}"
}

run_dapo_ablation() {
  # 1) baseline GRPO clip
  run_case "${GROUP}_dapo00_baseline_grpo_clip_seq" "grpo_clip" "seq" "--no-use_std_normalization" \
    "batch,anypo,baseline,grpo_clip,seq,dapo_ablation" \
    --no-use_kl_penalty

  # 2) decoupled clip only
  run_case "${GROUP}_dapo01_decoupled_clip_seq" "dapo_clip" "seq" "--no-use_std_normalization" \
    "batch,anypo,dapo,decoupled_clip,seq,dapo_ablation" \
    --cliprange_low "${CLIP_LOW:-0.2}" \
    --cliprange_high "${CLIP_HIGH:-0.28}" \
    --no-enable_group_filter \
    --no-use_overlong_penalty \
    --no-use_kl_penalty

  # 3) +group filter
  run_case "${GROUP}_dapo02_decoupled_clip_filter_seq" "dapo_clip" "seq" "--no-use_std_normalization" \
    "batch,anypo,dapo,decoupled_clip,group_filter,seq,dapo_ablation" \
    --cliprange_low "${CLIP_LOW:-0.2}" \
    --cliprange_high "${CLIP_HIGH:-0.28}" \
    --enable_group_filter \
    --filter_all_zero \
    --filter_all_one \
    --max_filter_resample_rounds "${MAX_FILTER_RESAMPLE_ROUNDS:-8}" \
    --no-use_overlong_penalty \
    --no-use_kl_penalty

  # 4) +overlong penalty
  run_case "${GROUP}_dapo03_decoupled_clip_overlong_seq" "dapo_clip" "seq" "--no-use_std_normalization" \
    "batch,anypo,dapo,decoupled_clip,overlong,seq,dapo_ablation" \
    --cliprange_low "${CLIP_LOW:-0.2}" \
    --cliprange_high "${CLIP_HIGH:-0.28}" \
    --no-enable_group_filter \
    --use_overlong_penalty \
    --overlong_target_len "${OVERLONG_TARGET_LEN:-256}" \
    --overlong_penalty_coef "${OVERLONG_PENALTY_COEF:-0.02}" \
    --no-use_kl_penalty

  # 5) full dapo-lite
  run_case "${GROUP}_dapo04_full_seq" "dapo_clip" "seq" "--no-use_std_normalization" \
    "batch,anypo,dapo,full,seq,dapo_ablation" \
    --cliprange_low "${CLIP_LOW:-0.2}" \
    --cliprange_high "${CLIP_HIGH:-0.28}" \
    --enable_group_filter \
    --filter_all_zero \
    --filter_all_one \
    --max_filter_resample_rounds "${MAX_FILTER_RESAMPLE_ROUNDS:-8}" \
    --use_overlong_penalty \
    --overlong_target_len "${OVERLONG_TARGET_LEN:-256}" \
    --overlong_penalty_coef "${OVERLONG_PENALTY_COEF:-0.02}" \
    --no-use_kl_penalty
}

run_gbpo_ablation() {
  # 1) baseline GRPO clip seq
  run_case "${GROUP}_gbpo00_baseline_grpo_clip_seq" "grpo_clip" "seq" "--no-use_std_normalization" \
    "batch,anypo,baseline,grpo_clip,seq,gbpo_ablation" \
    --no-use_kl_penalty

  # 2) gbpo sign-aware clip seq
  run_case "${GROUP}_gbpo01_signaware_seq" "gbpo_clip" "seq" "--no-use_std_normalization" \
    "batch,anypo,gbpo,signaware,seq,gbpo_ablation" \
    --gbpo_pos_cliprange_low "${GBPO_POS_LOW:-0.2}" \
    --gbpo_pos_cliprange_high "${GBPO_POS_HIGH:-0.28}" \
    --gbpo_neg_cliprange_low "${GBPO_NEG_LOW:-0.05}" \
    --gbpo_neg_cliprange_high "${GBPO_NEG_HIGH:-0.10}" \
    --no-use_kl_penalty

  # 3) gbpo sign-aware clip seq + KL
  run_case "${GROUP}_gbpo02_signaware_seq_kl" "gbpo_clip" "seq" "--no-use_std_normalization" \
    "batch,anypo,gbpo,signaware,seq,kl,gbpo_ablation" \
    --gbpo_pos_cliprange_low "${GBPO_POS_LOW:-0.2}" \
    --gbpo_pos_cliprange_high "${GBPO_POS_HIGH:-0.28}" \
    --gbpo_neg_cliprange_low "${GBPO_NEG_LOW:-0.05}" \
    --gbpo_neg_cliprange_high "${GBPO_NEG_HIGH:-0.10}" \
    --use_kl_penalty \
    --kl_beta "${KL_BETA:-0.02}" \
    --kl_estimator "${KL_ESTIMATOR:-k3}"

  # 4) gbpo sign-aware clip token
  run_case "${GROUP}_gbpo03_signaware_token" "gbpo_clip" "token" "--no-use_std_normalization" \
    "batch,anypo,gbpo,signaware,token,gbpo_ablation" \
    --gbpo_pos_cliprange_low "${GBPO_POS_LOW:-0.2}" \
    --gbpo_pos_cliprange_high "${GBPO_POS_HIGH:-0.28}" \
    --gbpo_neg_cliprange_low "${GBPO_NEG_LOW:-0.05}" \
    --gbpo_neg_cliprange_high "${GBPO_NEG_HIGH:-0.10}" \
    --no-use_kl_penalty
}

if [[ "$MODE" == "token_fill" ]]; then
  run_token_fill
elif [[ "$MODE" == "kl_ablation" ]]; then
  run_kl_ablation
elif [[ "$MODE" == "dapo_ablation" ]]; then
  run_dapo_ablation
elif [[ "$MODE" == "gbpo_ablation" ]]; then
  run_gbpo_ablation
else
  run_full_or_smoke
fi

echo
echo "Batch finished."
echo "W&B project: $PROJECT"
echo "W&B group:   $GROUP"
echo "Save root:   $SAVE_ROOT"
