# any_po

统一测试 AnyPO / GRPO 变体的实验仓库。

## 训练入口

主入口脚本：
- `cs336_alignment/train_anypo.py`
- `scripts/run_anypo_batch.sh`（支持 baseline / KL / DAPO / GBPO 批量）

核心算法实现：
- `cs336_alignment/anypo.py`

## 常用参数

- `--loss_type`: `no_baseline | reinforce_with_baseline | grpo_no_clip | grpo_clip | dapo_clip | gbpo_clip`
- `--use_std_normalization / --no-use_std_normalization`
- `--loss_aggregation`: `seq | token`
- `--use_kl_penalty / --no-use_kl_penalty`
- `--kl_beta`
- `--kl_estimator`: `k1 | k2 | k3`（默认 `k3`）
- DAPO clip：`--cliprange_low --cliprange_high`
- GBPO clip：`--gbpo_pos_cliprange_low --gbpo_pos_cliprange_high --gbpo_neg_cliprange_low --gbpo_neg_cliprange_high`
- group filter：`--enable_group_filter --filter_all_zero --filter_all_one --max_filter_resample_rounds`
- overlong penalty：`--use_overlong_penalty --overlong_target_len --overlong_penalty_coef`
- `--seed`
- `--save_dir`（可选，保存 `best/final/step_xxx`）

示例：

```bash
uv run python cs336_alignment/train_anypo.py \
  --loss_type grpo_clip \
  --no-use_std_normalization \
  --loss_aggregation seq \
  --no-use_kl_penalty \
  --seed 69 \
  --project cs336-anypo \
  --run_name anypo_smoke \
  --save_dir /home/bkzhu/storage/assignment5-alignment/data/anypo_ckpts
```

说明：`use_kl_penalty` 默认关闭。也就是说，老实验命令不需要任何修改，行为与之前一致。

## 第一批实验组（不做 cliprange 扫描）

按你的最新要求，先不做 `cliprange` 变动，留到 DAPO（上下限 decoupled clip）阶段。

1. `grpo_clip + std_off + seq`
2. `grpo_no_clip + std_off + seq`
3. `reinforce_with_baseline + std_off + seq`
4. `no_baseline + std_off + seq`
5. `grpo_clip + std_on + seq`
6. `grpo_no_clip + std_on + seq`
7. `grpo_clip + std_off + token`
8. `grpo_no_clip + std_off + token`

建议先做一次 smoke（`--n_grpo_steps 4`）再跑全量。

## 批量运行

```bash
# 先跑 smoke
bash scripts/run_anypo_batch.sh smoke

# 再跑 full
bash scripts/run_anypo_batch.sh full

# 仅补跑缺失 token 组
bash scripts/run_anypo_batch.sh token_fill

# KL 对比（no-KL vs +KL）
bash scripts/run_anypo_batch.sh kl_ablation

# DAPO-lite 消融
bash scripts/run_anypo_batch.sh dapo_ablation

# GBPO-lite 消融
bash scripts/run_anypo_batch.sh gbpo_ablation
```

可用环境变量：
- `WANDB_PROJECT`
- `WANDB_GROUP`
- `SAVE_ROOT`
- `SEED`
- `DEVICE_TRAIN`
- `DEVICE_VLLM`
- `N_GRPO_STEPS`（覆盖 smoke/full 默认）
- `KL_BETA`（仅 `kl_ablation` 模式）
- `KL_ESTIMATOR`（仅 `kl_ablation` 模式，`k1|k2|k3`）
- `CLIP_LOW` / `CLIP_HIGH`（`dapo_ablation`）
- `MAX_FILTER_RESAMPLE_ROUNDS`（`dapo_ablation`）
- `OVERLONG_TARGET_LEN` / `OVERLONG_PENALTY_COEF`（`dapo_ablation`）
- `GBPO_POS_LOW` / `GBPO_POS_HIGH` / `GBPO_NEG_LOW` / `GBPO_NEG_HIGH`（`gbpo_ablation`）

说明：批量脚本会自动把每个 run 保存到独立目录：`$SAVE_ROOT/$run_name`，不会互相覆盖。

## 统一实验表（建议按这个对照看）

| 模式 | run_name 后缀 | 主要改动 | 主要对照对象 | 重点看哪些指标 |
| --- | --- | --- | --- | --- |
| `kl_ablation` | `kl00_clip_token_std0_no_kl` | `grpo_clip + token`，不加 KL | `kl01_clip_token_std0_with_kl` | `train/avg_kl`、`train/avg_kl_loss`、`eval/accuracy` |
| `kl_ablation` | `kl01_clip_token_std0_with_kl` | `grpo_clip + token + KL(k3,beta)` | `kl00_clip_token_std0_no_kl` | 同上，额外看 `train/avg_pg_loss` |
| `kl_ablation` | `kl00_clip_seq_std0_no_kl` | `grpo_clip + seq`，不加 KL | `kl01_clip_seq_std0_with_kl` | `train/loss`、`train/grad_norm`、`eval/accuracy` |
| `kl_ablation` | `kl01_clip_seq_std0_with_kl` | `grpo_clip + seq + KL(k3,beta)` | `kl00_clip_seq_std0_no_kl` | 同上，额外看 `train/avg_kl` |
| `dapo_ablation` | `dapo00_baseline_grpo_clip_seq` | 纯 `grpo_clip + seq` baseline | `dapo01/02/03/04` | `eval/accuracy`、`train/avg_clip_fraction` |
| `dapo_ablation` | `dapo01_decoupled_clip_seq` | `loss_type=dapo_clip`（上下限分离 clip） | `dapo00` | `train/clip_low_hit_rate`、`train/clip_high_hit_rate` |
| `dapo_ablation` | `dapo02_decoupled_clip_filter_seq` | `dapo_clip + group_filter` | `dapo01` | `sampling/valid_group_ratio`、`sampling/resample_count` |
| `dapo_ablation` | `dapo03_decoupled_clip_overlong_seq` | `dapo_clip + overlong_penalty` | `dapo01` | `sampling/overlong_penalty_mean`、`eval/avg_length` |
| `dapo_ablation` | `dapo04_full_seq` | `dapo_clip + group_filter + overlong_penalty` | `dapo00`、`dapo02`、`dapo03` | `eval/accuracy`、`train/grad_norm`、长度相关指标 |
| `gbpo_ablation` | `gbpo00_baseline_grpo_clip_seq` | 纯 `grpo_clip + seq` baseline | `gbpo01/02/03` | `eval/accuracy`、`train/grad_norm` |
| `gbpo_ablation` | `gbpo01_signaware_seq` | `loss_type=gbpo_clip`（正负优势分开 clip） | `gbpo00` | `train/gbpo_pos_clip_hit_rate`、`train/gbpo_neg_clip_hit_rate` |
| `gbpo_ablation` | `gbpo02_signaware_seq_kl` | `gbpo_clip + KL` | `gbpo01` | `train/avg_kl`、`train/avg_kl_loss`、`eval/accuracy` |
| `gbpo_ablation` | `gbpo03_signaware_token` | `gbpo_clip + token aggregation` | `gbpo01` | `train/loss`、`train/avg_token_entropy`、`eval/accuracy` |

备注：
- 所有批量实验默认固定 `std_off`，方便先观察核心 loss 改动。
- `KL_BETA`、`KL_ESTIMATOR`、`CLIP_LOW/HIGH`、`GBPO_*` 可用环境变量覆盖。
- 推荐先比较同一模式内的“相邻对照”（例如 `dapo01 -> dapo02`），再比较跨模式最佳点。

## W&B 细粒度日志

训练入口会记录：
- 详细配置：完整 `AnyPOConfig` + 派生配置（micro-batch、rollout prompts 数、hostname）
- 运行结构化字段：`group / job_type / tags / notes`
- 双步轴：
  - `grpo_step`（采样与评估）
  - `train/update_step`（每次 optimizer step）
- 采样统计：
  - reward 均值/方差/最小/最大
  - reward_rate / format_rate / answer_rate
  - all-negative / all-positive group 比例
  - response 长度分布
  - advantage 均值/方差/绝对值均值
- 训练统计：
  - loss / grad_norm / lr
  - avg_token_entropy / avg_clip_fraction
  - mean_response_length
  - KL 打开时：`avg_kl / avg_kl_loss / avg_pg_loss / kl_beta`
- 评估统计：
  - accuracy / correct / format-wrong / answer-wrong / avg_length

## KL 消融（可直接跑）

在同一组配置上做 `+KL` 对比（建议先做在 `grpo_clip + seq + std_off`）：

```bash
# baseline (no KL)
uv run python cs336_alignment/train_anypo.py \
  --loss_type grpo_clip \
  --loss_aggregation seq \
  --no-use_std_normalization \
  --no-use_kl_penalty \
  --run_name clip_seq_std0_no_kl

# +KL
uv run python cs336_alignment/train_anypo.py \
  --loss_type grpo_clip \
  --loss_aggregation seq \
  --no-use_std_normalization \
  --use_kl_penalty \
  --kl_beta 0.02 \
  --kl_estimator k3 \
  --run_name clip_seq_std0_with_kl
```

## DAPO / GBPO 运行示例

```bash
# DAPO-lite（单实验）
uv run python cs336_alignment/train_anypo.py \
  --loss_type dapo_clip \
  --loss_aggregation seq \
  --cliprange_low 0.2 \
  --cliprange_high 0.28 \
  --enable_group_filter \
  --filter_all_zero \
  --filter_all_one \
  --max_filter_resample_rounds 8 \
  --use_overlong_penalty \
  --overlong_target_len 256 \
  --overlong_penalty_coef 0.02 \
  --run_name dapo_lite_seq

# GBPO-lite（单实验）
uv run python cs336_alignment/train_anypo.py \
  --loss_type gbpo_clip \
  --loss_aggregation seq \
  --gbpo_pos_cliprange_low 0.2 \
  --gbpo_pos_cliprange_high 0.28 \
  --gbpo_neg_cliprange_low 0.05 \
  --gbpo_neg_cliprange_high 0.10 \
  --run_name gbpo_lite_seq
```
