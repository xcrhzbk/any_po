#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-smoke}" # smoke | full | token_fill | kl_ablation | dapo_ablation | gbpo_ablation | step_ablation
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

PYTHON_CMD="${PYTHON_CMD:-uv run python}"
PROMPT_PATH="${PROMPT_PATH:-$ROOT_DIR/cs336_alignment/prompts/r1_zero.prompt}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$ROOT_DIR/data/gsm8k/train.jsonl}"
if [[ -n "${TEST_DATA_PATH:-}" ]]; then
  TEST_DATA_PATH="$TEST_DATA_PATH"
else
  TRAIN_DIR="$(dirname "$TRAIN_DATA_PATH")"
  AUTO_TEST_PATH="${TRAIN_DIR}/test.jsonl"
  if [[ -f "$AUTO_TEST_PATH" ]]; then
    TEST_DATA_PATH="$AUTO_TEST_PATH"
  else
    TEST_DATA_PATH="$ROOT_DIR/data/gsm8k/test.jsonl"
  fi
fi

DATASET_TAG="${DATASET_TAG:-$(basename "$(dirname "$TRAIN_DATA_PATH")")}"
PROJECT="${WANDB_PROJECT:-any3}"
GROUP="${WANDB_GROUP:-any3_${DATASET_TAG}_${MODE}_${TIMESTAMP}}"
SAVE_ROOT="${SAVE_ROOT:-$ROOT_DIR/data/anypo_ckpts/${GROUP}}"

SEED="${SEED:-69}"
DEVICE_TRAIN="${DEVICE_TRAIN:-cuda:0}"
DEVICE_VLLM="${DEVICE_VLLM:-cuda:1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
GROUP_SIZE="${GROUP_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-64}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.3}"
EPOCHS_PER_ROLLOUT="${EPOCHS_PER_ROLLOUT:-3}"
TARGET_UPDATES="${TARGET_UPDATES:-300}"

if (( ROLLOUT_BATCH_SIZE % TRAIN_BATCH_SIZE != 0 )); then
  echo "ERROR: ROLLOUT_BATCH_SIZE must be divisible by TRAIN_BATCH_SIZE"
  exit 1
fi
UPDATES_PER_GRPO_STEP=$(( (ROLLOUT_BATCH_SIZE / TRAIN_BATCH_SIZE) * EPOCHS_PER_ROLLOUT ))
if (( UPDATES_PER_GRPO_STEP <= 0 )); then
  echo "ERROR: invalid UPDATES_PER_GRPO_STEP=$UPDATES_PER_GRPO_STEP"
  exit 1
fi

if [[ "$MODE" == "smoke" ]]; then
  N_GRPO_STEPS="${N_GRPO_STEPS:-4}"
  EVAL_FREQ="${EVAL_FREQ:-2}"
  EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-128}"
else
  # Keep total optimizer updates around TARGET_UPDATES while using off-policy epochs.
  DEFAULT_GRPO_STEPS=$(( (TARGET_UPDATES + UPDATES_PER_GRPO_STEP - 1) / UPDATES_PER_GRPO_STEP ))
  N_GRPO_STEPS="${N_GRPO_STEPS:-$DEFAULT_GRPO_STEPS}"
  EVAL_FREQ="${EVAL_FREQ:-8}"
  EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-0}"
fi

COMMON_ARGS=(
  --project "$PROJECT"
  --prompt_path "$PROMPT_PATH"
  --train_data_path "$TRAIN_DATA_PATH"
  --test_data_path "$TEST_DATA_PATH"
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
  --epochs_per_rollout_batch "$EPOCHS_PER_ROLLOUT"
  --gradient_accumulation_steps "$GRAD_ACCUM"
  --save_best
)

slug_value() {
  local raw="$1"
  raw="${raw,,}"
  raw="${raw//./p}"
  raw="${raw//[^a-z0-9_-]/-}"
  echo "$raw"
}

build_run_config_tag() {
  local loss_type="$1"
  local loss_aggregation="$2"
  local std_flag="$3"
  shift 3
  local extra_args=("$@")

  local std_tag="std0"
  if [[ "$std_flag" == "--use_std_normalization" ]]; then
    std_tag="std1"
  fi

  local kl_on=0
  local kl_beta="0"
  local kl_est="none"

  local clip_low=""
  local clip_high=""
  local gbpo_pos_low=""
  local gbpo_pos_high=""
  local gbpo_neg_low=""
  local gbpo_neg_high=""

  local gf="na"
  local gf_rounds=""

  local ol_on=0
  local ol_target=""
  local ol_coef=""

  local sp_on=0
  local sp_mode=""
  local sp_alpha=""
  local sp_m=""
  local sp_k=""

  local i=0
  while (( i < ${#extra_args[@]} )); do
    local arg="${extra_args[$i]}"
    case "$arg" in
      --use_kl_penalty)
        kl_on=1
        ;;
      --no-use_kl_penalty)
        kl_on=0
        ;;
      --kl_beta)
        ((i++))
        kl_beta="${extra_args[$i]}"
        ;;
      --kl_estimator)
        ((i++))
        kl_est="${extra_args[$i]}"
        ;;
      --cliprange_low)
        ((i++))
        clip_low="${extra_args[$i]}"
        ;;
      --cliprange_high)
        ((i++))
        clip_high="${extra_args[$i]}"
        ;;
      --gbpo_pos_cliprange_low)
        ((i++))
        gbpo_pos_low="${extra_args[$i]}"
        ;;
      --gbpo_pos_cliprange_high)
        ((i++))
        gbpo_pos_high="${extra_args[$i]}"
        ;;
      --gbpo_neg_cliprange_low)
        ((i++))
        gbpo_neg_low="${extra_args[$i]}"
        ;;
      --gbpo_neg_cliprange_high)
        ((i++))
        gbpo_neg_high="${extra_args[$i]}"
        ;;
      --enable_group_filter)
        gf=1
        ;;
      --no-enable_group_filter)
        gf=0
        ;;
      --max_filter_resample_rounds)
        ((i++))
        gf_rounds="${extra_args[$i]}"
        ;;
      --use_overlong_penalty)
        ol_on=1
        ;;
      --no-use_overlong_penalty)
        ol_on=0
        ;;
      --overlong_target_len)
        ((i++))
        ol_target="${extra_args[$i]}"
        ;;
      --overlong_penalty_coef)
        ((i++))
        ol_coef="${extra_args[$i]}"
        ;;
      --use_step_progress_reward)
        sp_on=1
        ;;
      --no-use_step_progress_reward)
        sp_on=0
        ;;
      --step_reward_mode)
        ((i++))
        sp_mode="${extra_args[$i]}"
        ;;
      --step_reward_alpha)
        ((i++))
        sp_alpha="${extra_args[$i]}"
        ;;
      --step_reward_rollouts_per_prefix)
        ((i++))
        sp_m="${extra_args[$i]}"
        ;;
      --step_reward_max_steps)
        ((i++))
        sp_k="${extra_args[$i]}"
        ;;
    esac
    ((i++))
  done

  local kl_tag="kl0"
  if (( kl_on == 1 )); then
    kl_tag="kl1b$(slug_value "$kl_beta")k$(slug_value "$kl_est")"
  fi

  local algo_tag=""
  case "$loss_type" in
    dapo_clip)
      algo_tag="dapo"
      ;;
    gbpo_clip)
      algo_tag="gbpo"
      ;;
    grpo_clip|grpo_no_clip|reinforce_with_baseline|no_baseline)
      algo_tag="grpo"
      ;;
    *)
      algo_tag="$(slug_value "$loss_type")"
      ;;
  esac

  local parts=(
    "alg-${algo_tag}"
    "lt-$(slug_value "$loss_type")"
    "ag-$(slug_value "$loss_aggregation")"
    "$std_tag"
    "$kl_tag"
    "epr$(slug_value "$EPOCHS_PER_ROLLOUT")"
  )

  if [[ -n "$clip_low" && -n "$clip_high" ]]; then
    parts+=("cl$(slug_value "$clip_low")-$(slug_value "$clip_high")")
  fi
  if [[ -n "$gbpo_pos_low" && -n "$gbpo_pos_high" && -n "$gbpo_neg_low" && -n "$gbpo_neg_high" ]]; then
    parts+=("gbpp$(slug_value "$gbpo_pos_low")-$(slug_value "$gbpo_pos_high")")
    parts+=("gbpn$(slug_value "$gbpo_neg_low")-$(slug_value "$gbpo_neg_high")")
  fi
  if [[ "$gf" != "na" ]]; then
    local gf_tag="gf${gf}"
    if [[ -n "$gf_rounds" ]]; then
      gf_tag="${gf_tag}r$(slug_value "$gf_rounds")"
    fi
    parts+=("$gf_tag")
  fi
  if (( ol_on == 1 )); then
    local ol_tag="ol1"
    if [[ -n "$ol_target" ]]; then
      ol_tag="${ol_tag}t$(slug_value "$ol_target")"
    fi
    if [[ -n "$ol_coef" ]]; then
      ol_tag="${ol_tag}c$(slug_value "$ol_coef")"
    fi
    parts+=("$ol_tag")
  fi
  if (( sp_on == 1 )); then
    local sp_tag="sp1"
    if [[ -n "$sp_mode" ]]; then
      sp_tag="${sp_tag}m$(slug_value "$sp_mode")"
    fi
    if [[ -n "$sp_alpha" ]]; then
      sp_tag="${sp_tag}a$(slug_value "$sp_alpha")"
    fi
    if [[ -n "$sp_m" ]]; then
      sp_tag="${sp_tag}r$(slug_value "$sp_m")"
    fi
    if [[ -n "$sp_k" ]]; then
      sp_tag="${sp_tag}k$(slug_value "$sp_k")"
    fi
    parts+=("$sp_tag")
  fi

  local joined="${parts[*]}"
  echo "${joined// /_}"
}

run_case() {
  local run_alias="$1"
  local loss_type="$2"
  local loss_aggregation="$3"
  local std_flag="$4"
  local tags="$5"
  shift 5
  local extra_args=("$@")
  local config_tag
  config_tag="$(build_run_config_tag "$loss_type" "$loss_aggregation" "$std_flag" "${extra_args[@]}")"

  local case_suffix="$run_alias"
  if [[ "$case_suffix" == "${GROUP}_"* ]]; then
    case_suffix="${case_suffix#${GROUP}_}"
  fi
  local case_id="${case_suffix%%_*}"
  local run_name="${GROUP}_${case_id}_${config_tag}"
  if (( ${#run_name} > 220 )); then
    local run_hash
    run_hash="$(printf '%s' "$run_name" | cksum | awk '{print $1}')"
    run_name="${run_name:0:180}_h${run_hash}"
  fi

  echo "============================================================"
  echo "Running: $run_name"
  echo "  case_id=$case_id"
  echo "  case_suffix=$case_suffix"
  echo "  config_tag=$config_tag"
  echo "  loss_type=$loss_type"
  echo "  loss_aggregation=$loss_aggregation"
  echo "  std_flag=$std_flag"
  echo "  dataset_tag=$DATASET_TAG"
  echo "  train_data_path=$TRAIN_DATA_PATH"
  echo "  test_data_path=$TEST_DATA_PATH"
  echo "  epochs_per_rollout_batch=$EPOCHS_PER_ROLLOUT"
  echo "  updates_per_grpo_step=$UPDATES_PER_GRPO_STEP"
  echo "  target_updates=$TARGET_UPDATES"
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
    --wandb_notes "batch=${GROUP};dataset=${DATASET_TAG};mode=${MODE};case=${case_suffix};config=${config_tag};loss=${loss_type};agg=${loss_aggregation};std=${std_flag};epochs_per_rollout=${EPOCHS_PER_ROLLOUT};updates_per_step=${UPDATES_PER_GRPO_STEP};target_updates=${TARGET_UPDATES}" \
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

run_step_ablation() {
  local STEP_LAMBDA="${STEP_LAMBDA:-0.95}"
  local STEP_GAMMA="${STEP_GAMMA:-1.0}"
  local STEP_MIN_CHARS="${STEP_MIN_CHARS:-24}"
  local STEP_EVAL_BATCH="${STEP_EVAL_BATCH:-64}"
  local STEP_MAX_SELECTED="${STEP_MAX_SELECTED:-64}"
  local STEP_MAX_TOKENS="${STEP_MAX_TOKENS:-256}"

  # 0) baseline
  run_case "${GROUP}_sp00_baseline_grpo_clip_seq" "grpo_clip" "seq" "--no-use_std_normalization" \
    "batch,anypo,step_progress,baseline,seq,ablation" \
    --no-use_step_progress_reward \
    --no-use_kl_penalty

  # 1) step-progress on all samples (m=2, k=4)
  run_case "${GROUP}_sp01_progress_all_m2_k4_a03" "grpo_clip" "seq" "--no-use_std_normalization" \
    "batch,anypo,step_progress,all,m2,k4,a03,ablation" \
    --use_step_progress_reward \
    --step_reward_mode all \
    --step_reward_alpha "${STEP_ALPHA_ALL:-0.3}" \
    --step_reward_rollouts_per_prefix 2 \
    --step_reward_max_steps 4 \
    --step_reward_lambda "$STEP_LAMBDA" \
    --step_reward_gamma "$STEP_GAMMA" \
    --step_reward_min_chars "$STEP_MIN_CHARS" \
    --step_reward_eval_batch_size "$STEP_EVAL_BATCH" \
    --step_reward_max_selected_samples "$STEP_MAX_SELECTED" \
    --step_reward_sampling_max_tokens "$STEP_MAX_TOKENS" \
    --no-use_kl_penalty

  # 2) step-progress on all samples (m=4, k=4)
  run_case "${GROUP}_sp02_progress_all_m4_k4_a03" "grpo_clip" "seq" "--no-use_std_normalization" \
    "batch,anypo,step_progress,all,m4,k4,a03,ablation" \
    --use_step_progress_reward \
    --step_reward_mode all \
    --step_reward_alpha "${STEP_ALPHA_ALL:-0.3}" \
    --step_reward_rollouts_per_prefix 4 \
    --step_reward_max_steps 4 \
    --step_reward_lambda "$STEP_LAMBDA" \
    --step_reward_gamma "$STEP_GAMMA" \
    --step_reward_min_chars "$STEP_MIN_CHARS" \
    --step_reward_eval_batch_size "$STEP_EVAL_BATCH" \
    --step_reward_max_selected_samples "$STEP_MAX_SELECTED" \
    --step_reward_sampling_max_tokens "$STEP_MAX_TOKENS" \
    --no-use_kl_penalty

  # 3) step-progress only on all-negative groups (m=2, k=4)
  run_case "${GROUP}_sp03_progress_negonly_m2_k4_a03" "grpo_clip" "seq" "--no-use_std_normalization" \
    "batch,anypo,step_progress,neg_only,m2,k4,a03,ablation" \
    --use_step_progress_reward \
    --step_reward_mode neg_only \
    --step_reward_alpha "${STEP_ALPHA_NEG:-0.3}" \
    --step_reward_rollouts_per_prefix 2 \
    --step_reward_max_steps 4 \
    --step_reward_lambda "$STEP_LAMBDA" \
    --step_reward_gamma "$STEP_GAMMA" \
    --step_reward_min_chars "$STEP_MIN_CHARS" \
    --step_reward_eval_batch_size "$STEP_EVAL_BATCH" \
    --step_reward_max_selected_samples "$STEP_MAX_SELECTED" \
    --step_reward_sampling_max_tokens "$STEP_MAX_TOKENS" \
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
elif [[ "$MODE" == "step_ablation" ]]; then
  run_step_ablation
else
  run_full_or_smoke
fi

echo
echo "Batch finished."
echo "W&B project: $PROJECT"
echo "W&B group:   $GROUP"
echo "Save root:   $SAVE_ROOT"
