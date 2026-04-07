#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SUITE="${1:-grpo_core}" # grpo_core | step_policy | policy_ext(alias) | all | smoke
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

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
GROUP_BASE="${WANDB_GROUP_BASE:-any3_${SUITE}_${DATASET_TAG}_${TIMESTAMP}}"
SAVE_ROOT_BASE="${SAVE_ROOT_BASE:-$ROOT_DIR/data/anypo_ckpts/${GROUP_BASE}}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"

case "$SUITE" in
  grpo_core)
    MODES=(full token_fill kl_ablation)
    ;;
  step_policy)
    MODES=(step_ablation dapo_ablation gbpo_ablation)
    ;;
  policy_ext)
    # Backward-compatible alias.
    MODES=(step_ablation dapo_ablation gbpo_ablation)
    ;;
  all)
    MODES=(full token_fill kl_ablation dapo_ablation gbpo_ablation step_ablation)
    ;;
  smoke)
    MODES=(smoke token_fill kl_ablation)
    export N_GRPO_STEPS="${N_GRPO_STEPS:-4}"
    export EVAL_FREQ="${EVAL_FREQ:-2}"
    export EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-128}"
    ;;
  *)
    echo "Unknown suite: $SUITE"
    echo "Valid suites: grpo_core | step_policy | policy_ext | all | smoke"
    exit 1
    ;;
esac

mkdir -p "$SAVE_ROOT_BASE" "$LOG_DIR"

echo "============================================================"
echo "One-shot AnyPO suite launcher"
echo "  suite:        $SUITE"
echo "  project:      $PROJECT"
echo "  dataset_tag:  $DATASET_TAG"
echo "  train_data:   $TRAIN_DATA_PATH"
echo "  test_data:    $TEST_DATA_PATH"
echo "  group_base:   $GROUP_BASE"
echo "  save_root:    $SAVE_ROOT_BASE"
echo "  modes:        ${MODES[*]}"
echo "============================================================"
echo

for MODE in "${MODES[@]}"; do
  MODE_GROUP="${GROUP_BASE}_${MODE}"
  MODE_SAVE_ROOT="${SAVE_ROOT_BASE}/${MODE}"
  MODE_LOG="${LOG_DIR}/${MODE_GROUP}.log"

  echo ">>> Launching mode: $MODE"
  echo "    WANDB_GROUP=$MODE_GROUP"
  echo "    SAVE_ROOT=$MODE_SAVE_ROOT"
  echo "    LOG=$MODE_LOG"

  WANDB_PROJECT="$PROJECT" \
  WANDB_GROUP="$MODE_GROUP" \
  TRAIN_DATA_PATH="$TRAIN_DATA_PATH" \
  TEST_DATA_PATH="$TEST_DATA_PATH" \
  SAVE_ROOT="$MODE_SAVE_ROOT" \
  bash scripts/run_anypo_batch.sh "$MODE" 2>&1 | tee "$MODE_LOG"

  echo ">>> Mode done: $MODE"
  echo
done

echo "All suite modes finished."
