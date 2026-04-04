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
- final eval 输出：`--save_final_eval_outputs --final_eval_output_max_samples --final_eval_output_filename`
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
另外 AnyPO 在最后一个评估 step 会默认保存测试样本输出（默认 100 条）到：
`$save_dir/final_eval_outputs.jsonl`。

## Math12k 数据准备

已提供一键脚本：

```bash
uv run python cs336_alignment/prepare_math12k.py
```

会生成：
- `data/math12k/train.jsonl`（AnyPO/GRPO 用，字段 `question/answer`）
- `data/math12k/test.jsonl`
- `data/math12k/processed_train.jsonl`（EI 用，字段 `prompt/response`）

使用示例：

```bash
# AnyPO / GRPO 跑 math12k
uv run python cs336_alignment/train_anypo.py \
  --train_data_path /home/bkzhu/storage/assignment5-alignment/data/math12k/train.jsonl \
  --test_data_path /home/bkzhu/storage/assignment5-alignment/data/math12k/test.jsonl

# EI 跑 math12k
uv run python cs336_alignment/ei.py \
  --train_data_path /home/bkzhu/storage/assignment5-alignment/data/math12k/processed_train.jsonl \
  --test_data_path /home/bkzhu/storage/assignment5-alignment/data/math12k/test.jsonl
```

## 统一实验表（完整可跑清单）

命名规则：
- 实际 run 名是 `${GROUP}_<suffix>`，下面表格列的是 `<suffix>`
- `smoke` 和 `full` 跑的是同一批 suffix（只是步数不同）

### A. baseline / token 组（12 组）

| 模式 | suffix | loss_type | agg | std | 说明 |
| --- | --- | --- | --- | --- | --- |
| `smoke/full` | `01_clip_seq_std0` | `grpo_clip` | `seq` | off | baseline |
| `smoke/full` | `02_noclip_seq_std0` | `grpo_no_clip` | `seq` | off | 去掉 clip |
| `smoke/full` | `03_reinforce_seq_std0` | `reinforce_with_baseline` | `seq` | off | REINFORCE+baseline |
| `smoke/full` | `04_nobase_seq_std0` | `no_baseline` | `seq` | off | 无 baseline |
| `smoke/full` | `05_clip_seq_std1` | `grpo_clip` | `seq` | on | 开 std norm |
| `smoke/full` | `06_noclip_seq_std1` | `grpo_no_clip` | `seq` | on | no clip + std |
| `smoke/full` | `07_clip_token_std0` | `grpo_clip` | `token` | off | token 聚合 |
| `smoke/full` | `08_noclip_token_std0` | `grpo_no_clip` | `token` | off | token + no clip |
| `token_fill` | `09_reinforce_token_std0` | `reinforce_with_baseline` | `token` | off | 补齐 token 对照 |
| `token_fill` | `10_nobase_token_std0` | `no_baseline` | `token` | off | 补齐 token 对照 |
| `token_fill` | `11_clip_token_std1` | `grpo_clip` | `token` | on | token + std |
| `token_fill` | `12_noclip_token_std1` | `grpo_no_clip` | `token` | on | token + no clip + std |

### B. KL 消融（4 组）

| 模式 | suffix | loss_type | agg | 额外开关 | 主要对照 |
| --- | --- | --- | --- | --- | --- |
| `kl_ablation` | `kl00_clip_token_std0_no_kl` | `grpo_clip` | `token` | `--no-use_kl_penalty` | vs `kl01_*_token_*` |
| `kl_ablation` | `kl01_clip_token_std0_with_kl` | `grpo_clip` | `token` | `--use_kl_penalty --kl_beta --kl_estimator` | vs `kl00_*_token_*` |
| `kl_ablation` | `kl00_clip_seq_std0_no_kl` | `grpo_clip` | `seq` | `--no-use_kl_penalty` | vs `kl01_*_seq_*` |
| `kl_ablation` | `kl01_clip_seq_std0_with_kl` | `grpo_clip` | `seq` | `--use_kl_penalty --kl_beta --kl_estimator` | vs `kl00_*_seq_*` |

### C. DAPO-lite 消融（5 组）

| 模式 | suffix | loss_type | 核心开关 | 主要对照 |
| --- | --- | --- | --- | --- |
| `dapo_ablation` | `dapo00_baseline_grpo_clip_seq` | `grpo_clip` | baseline | vs `dapo01~04` |
| `dapo_ablation` | `dapo01_decoupled_clip_seq` | `dapo_clip` | `cliprange_low/high` | vs `dapo00` |
| `dapo_ablation` | `dapo02_decoupled_clip_filter_seq` | `dapo_clip` | `+enable_group_filter` | vs `dapo01` |
| `dapo_ablation` | `dapo03_decoupled_clip_overlong_seq` | `dapo_clip` | `+use_overlong_penalty` | vs `dapo01` |
| `dapo_ablation` | `dapo04_full_seq` | `dapo_clip` | `decoupled_clip + filter + overlong` | vs `dapo00/02/03` |

### D. GBPO-lite 消融（4 组）

| 模式 | suffix | loss_type | 核心开关 | 主要对照 |
| --- | --- | --- | --- | --- |
| `gbpo_ablation` | `gbpo00_baseline_grpo_clip_seq` | `grpo_clip` | baseline | vs `gbpo01~03` |
| `gbpo_ablation` | `gbpo01_signaware_seq` | `gbpo_clip` | `gbpo_pos_* + gbpo_neg_*` | vs `gbpo00` |
| `gbpo_ablation` | `gbpo02_signaware_seq_kl` | `gbpo_clip` | `gbpo + KL` | vs `gbpo01` |
| `gbpo_ablation` | `gbpo03_signaware_token` | `gbpo_clip` | `gbpo + token aggregation` | vs `gbpo01` |

重点建议：
- 先看同模式内相邻对照（例如 `dapo01 -> dapo02`）
- 再对比各模式最佳点

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
一次性全跑
```bash 
cd /home/bkzhu/storage/assignment5-alignment

GROUP="grpo_all_$(date +%Y%m%d_%H%M%S)"
export WANDB_PROJECT="cs336-anypo2"
export WANDB_GROUP="$GROUP"
export SAVE_ROOT="/home/bkzhu/storage/assignment5-alignment/data/anypo_ckpts/$GROUP"

# 可选：你的设备配置
export DEVICE_TRAIN="cuda:0"
export DEVICE_VLLM="cuda:1"

for MODE in full token_fill kl_ablation; do
  bash scripts/run_anypo_batch.sh "$MODE"
done
```
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

## EI（阈值筛题 + 断点续训）

`cs336_alignment/ei.py` 现在支持：
- 全训练集 pass-rate 评估后筛题：`selection_mode=static|dynamic`
- 阈值：`--train_pass_threshold`
- checkpoint：每个 EI step 自动保存到 `$output_path/checkpoints/ei_step_xxxx`
- 续训：`--resume_from` 或 `--auto_resume`

示例：

```bash
# 静态筛题：训练前在全训练集上评估一次，保留 pass-rate >= 0.5 的题
uv run python cs336_alignment/ei.py \
  --selection_mode static \
  --train_pass_threshold 0.5 \
  --selection_num_g 4 \
  --n_ei_steps 3 \
  --batch_size 4096 \
  --epochs 6

# 动态筛题：每个 EI step 前重新评估并筛题
uv run python cs336_alignment/ei.py \
  --selection_mode dynamic \
  --train_pass_threshold 0.5 \
  --selection_num_g 4 \
  --n_ei_steps 3

# 从上次中断处续训（自动找 latest checkpoint）
uv run python cs336_alignment/ei.py --auto_resume
```

```
uv run python cs336_alignment/ei.py \
  --auto_resume \
  --selection_mode dynamic \
  --train_pass_threshold 0.6
```
