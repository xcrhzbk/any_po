# any_po

做完了 cs336，不知道还能做什么？
cs336 作业 5 改参数太麻烦？

anypo，在 cs336 作业 5 的基础上，结合GRPO 相关改进（DAPO、GBPO、Step-Progress, GSPO,OPSD）和 EI 按难度动态筛题，提供不同 loss ，kl，on/off policy 配置，一键启动消融实验。


## 训练入口

主入口脚本：
- `cs336_alignment/train_anypo.py`
- `scripts/run_anypo_batch.sh`（支持 baseline / KL / DAPO / GBPO / Step-Progress 批量）
- `scripts/run_anypo_suite.sh`（总控脚本，一次性分 mode 启动整套实验）
- `cs336_alignment/train_opsd.py`
- `scripts/run_opsd_batch.sh`

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
- step-progress reward：
  - `--use_step_progress_reward --step_reward_mode`
  - `--step_reward_alpha --step_reward_gamma --step_reward_lambda`
  - `--step_reward_rollouts_per_prefix --step_reward_max_steps --step_reward_min_chars`
  - `--step_reward_eval_batch_size --step_reward_max_selected_samples`
  - `--step_reward_sampling_temperature --step_reward_sampling_max_tokens`
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
  --save_dir /home/<your path>/assignment5-alignment/data/anypo_ckpts
```

说明：`use_kl_penalty` 默认关闭。也就是说，老实验命令不需要任何修改，行为与之前一致。

## 推荐启动（覆盖优先）

新增总控脚本：`scripts/run_anypo_suite.sh`。  
它会用 `for mode in ...` 调用 `run_anypo_batch.sh`，并给每个 mode 分配独立 `WANDB_GROUP` 和 `SAVE_ROOT`，方便一次性跑完整套且分组对比。

| suite | 覆盖内容 | 总 run 数 |
| --- | --- | --- |
| `grpo_core` | `full + token_fill + kl_ablation` | 16 |
| `step_policy` | `step_ablation + dapo_ablation + gbpo_ablation` | 13 |
| `all` | `grpo_core + step_policy` | 29 |
| `smoke` | 快速冒烟（低步数） | 16 |

推荐直接用：

```bash
cd /home/<your path>/assignment5-alignment

export WANDB_PROJECT=any3
export WANDB_GROUP_BASE=anypo_all_$(date +%Y%m%d_%H%M%S)
export SAVE_ROOT_BASE=/home/<your path>/assignment5-alignment/data/anypo_ckpts/$WANDB_GROUP_BASE
export TRAIN_DATA_PATH=/home/<your path>/assignment5-alignment/data/math12k/train.jsonl
export TEST_DATA_PATH=/home/<your path>/assignment5-alignment/data/math12k/test.jsonl
export DEVICE_TRAIN=cuda:0
export DEVICE_VLLM=cuda:1
printf "TRAIN=%s\nTEST=%s\nDEV_TRAIN=%s\nDEV_VLLM=%s\nPROJECT=%s\n" \
  "$TRAIN_DATA_PATH" "$TEST_DATA_PATH" "$DEVICE_TRAIN" "$DEVICE_VLLM" "$WANDB_PROJECT" && \
test -f "$TRAIN_DATA_PATH" && test -f "$TEST_DATA_PATH" && \
echo "[OK] dataset paths exist" || { echo "[ERR] dataset path invalid"; exit 1; }

# 一次性跑完全部实验（按 mode 分组）
bash scripts/run_anypo_suite.sh all
```

只跑 RL 主线（推荐首选）：

```bash
bash scripts/run_anypo_suite.sh grpo_core
```

只跑 Step + Policy：

```bash
bash scripts/run_anypo_suite.sh step_policy
```

## 单模式批量（保留）

如果只想跑某一组：

```bash
bash scripts/run_anypo_batch.sh full
bash scripts/run_anypo_batch.sh token_fill
bash scripts/run_anypo_batch.sh kl_ablation
bash scripts/run_anypo_batch.sh dapo_ablation
bash scripts/run_anypo_batch.sh gbpo_ablation
bash scripts/run_anypo_batch.sh step_ablation
```

命名规则：
- 实际 run 名：`${WANDB_GROUP}_<case_id>_<config_tag>`
- `case_id` 是消融编号（如 `01` / `kl00` / `dapo04` / `sp02`），避免和配置签名重复
- `config_tag` 会自动编码关键配置：`alg/lt/ag/std/kl/epr`，并按需追加 `clip/gbpo/group_filter/overlong/step_reward`
- `run_anypo_suite.sh` 会自动把 `WANDB_GROUP` 设成 `${WANDB_GROUP_BASE}_${mode}`
- 每个 run 的 checkpoint 在：`$SAVE_ROOT/$run_name`
- 最后一轮评估样本输出默认在：`$save_dir/final_eval_outputs.jsonl`（默认 100 条）

消融 suffix（精简版）：
- `full/smoke`: `01~08`（GRPO 主干）
- `token_fill`: `09~12`（token 对照补齐）
- `kl_ablation`: `kl00/kl01`（token 与 seq 各一对）
- `dapo_ablation`: `dapo00~dapo04`
- `gbpo_ablation`: `gbpo00~gbpo03`
- `step_ablation`: `sp00~sp03`

## 关键环境变量

- 通用：`WANDB_PROJECT` `WANDB_GROUP/WANDB_GROUP_BASE` `SAVE_ROOT/SAVE_ROOT_BASE`
- 数据：`PROMPT_PATH` `TRAIN_DATA_PATH` `TEST_DATA_PATH`
- 设备：`DEVICE_TRAIN` `DEVICE_VLLM` `GPU_MEM_UTIL`
- 训练：`N_GRPO_STEPS` `EVAL_FREQ` `EVAL_NUM_SAMPLES` `EPOCHS_PER_ROLLOUT` `TARGET_UPDATES` `GRAD_ACCUM`
- KL：`KL_BETA` `KL_ESTIMATOR`
- DAPO：`CLIP_LOW` `CLIP_HIGH` `MAX_FILTER_RESAMPLE_ROUNDS` `OVERLONG_TARGET_LEN` `OVERLONG_PENALTY_COEF`
- GBPO：`GBPO_POS_LOW` `GBPO_POS_HIGH` `GBPO_NEG_LOW` `GBPO_NEG_HIGH`
- Step：`STEP_ALPHA_ALL` `STEP_ALPHA_NEG` `STEP_LAMBDA` `STEP_GAMMA` `STEP_MIN_CHARS` `STEP_EVAL_BATCH` `STEP_MAX_SELECTED` `STEP_MAX_TOKENS`

## Math12k 数据准备

```bash
uv run python cs336_alignment/prepare_math12k.py
```

输出：
- `data/math12k/train.jsonl`（AnyPO/GRPO）
- `data/math12k/test.jsonl`
- `data/math12k/processed_train.jsonl`（EI）

切换数据集（不改脚本）：

```bash
export TRAIN_DATA_PATH=/home/<your path>/assignment5-alignment/data/math12k/train.jsonl
export TEST_DATA_PATH=/home/<your path>/assignment5-alignment/data/math12k/test.jsonl
bash scripts/run_anypo_suite.sh grpo_core
```

## OPSD（On-Policy Self-Distillation）

新增：
- 训练入口：`cs336_alignment/train_opsd.py`
- 核心损失：`cs336_alignment/opsd.py`（response token 上的 KL distillation）
- 批量脚本：`scripts/run_opsd_batch.sh`

默认训练数据是 `train.jsonl`（`question/answer`），仅使用“标准答案”构造 teacher 的 privileged prompt。

一键批量（2 组）：

```bash
# 快速检查
bash scripts/run_opsd_batch.sh smoke

# 正式对比
bash scripts/run_opsd_batch.sh full
```

默认命名：
- `opsd01_gt_on`：使用标准答案增强 teacher prompt（论文口径的 OPSD）
- `opsd02_gt_off`：不加标准答案，仅自蒸馏控制组

核心差异（和你前面提到的 DPO/SDPO-pairwise 区分）：
- OPSD 是 token-level distillation：`KL(p_student || p_teacher)`，并且是 on-policy 轨迹
- 不是 pairwise chosen/rejected logistic loss

常用覆盖参数：
- `N_OPSD_STEPS`
- `PROMPT_BATCH_SIZE`
- `EPOCHS_PER_STEP`
- `KL_COEF`
- `TARGET_UPDATES`
- `TEACHER_USE_GROUND_TRUTH`（命令行参数：`--teacher_use_ground_truth / --no-teacher_use_ground_truth`）

## W&B 细粒度日志

默认训练策略说明：
- `run_anypo_batch.sh` 现在默认使用 `EPOCHS_PER_ROLLOUT=2`（off-policy 风格：同一批 rollout 多轮更新）
- 默认会根据 `TARGET_UPDATES=300` 自动换算 `N_GRPO_STEPS`，让总 optimizer updates 维持在约 300
- 若你要固定 `grpo_step=300`，手动设置 `N_GRPO_STEPS=300`

训练入口会记录：
- 详细配置：完整 `AnyPOConfig` + 派生配置（micro-batch、rollout prompts 数、hostname）
- 运行结构化字段：`group / job_type / tags / notes`
- 双步轴：`grpo_step` 与 `train/update_step`
- 采样统计：reward 分布、format/answer rate、group 统计、长度分布、adv 统计、step-progress 统计
- 训练统计：`loss / grad_norm / lr / avg_token_entropy / avg_clip_fraction / avg_kl / avg_kl_loss / avg_pg_loss`
- 评估统计：`accuracy / correct / format-wrong / answer-wrong / avg_length`

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

续训时可以改阈值，例如：

```bash
uv run python cs336_alignment/ei.py \
  --auto_resume \
  --selection_mode dynamic \
  --train_pass_threshold 0.6
```

```bash
# 阈值动态变化，逐渐降低每次 10 个 epoch
uv run python cs336_alignment/ei.py \
  --device_train cuda:4 \
  --device_vllm cuda:5 \
  --selection_mode dynamic \
  --train_pass_threshold 0.6 \
  --auto_threshold_decay \
  --threshold_end 0.0 \
  --n_ei_steps 7 \
  --epochs 10
```

## 方法原理总览（本仓库实现版）

### 1) `no_baseline`（原始 REINFORCE）

思路：直接用最终 reward 乘以 log-prob 做策略梯度更新。  
特点：实现最直接，但方差最大，训练曲线通常更抖。

### 2) `reinforce_with_baseline`

思路：把 reward 减去同组均值（baseline）得到 advantage，再做策略梯度。  
特点：不改目标期望，只降方差，稳定性比 `no_baseline` 好。

### 3) `grpo_no_clip`

思路：使用概率比值 `ratio = exp(logp_new - logp_old)` 的 PPO/GRPO 形式，但不做裁剪。  
特点：更新更激进，可能更快，但更容易出现 ratio 过大导致不稳定。

### 4) `grpo_clip`

思路：在 `grpo_no_clip` 基础上，对 ratio 做对称裁剪（`1 ± cliprange`）。  
特点：限制单步更新幅度，是当前主 baseline，通常速度与稳定性平衡较好。

### 5) `KL penalty`（可叠加在任意 loss）

思路：在策略损失外再加一项 `beta * KL(policy || ref)`（本仓库用 sampled-action 近似）。  
特点：抑制策略漂移，防止过拟合奖励模型；`beta` 越大约束越强。

### 6) DAPO-lite（`dapo_clip` + 采样策略）

实现由三块组成：
- Decoupled clip：上下界分离（`cliprange_low/high`），比对称 clip 更灵活。
- Group filter：过滤全 0 / 全 1 组，减少无信息或低信息 group。
- Overlong penalty：对超长输出加惩罚，抑制“靠拉长推理”取巧。

特点：更偏工程实用的稳定训练 recipe，通常在高步数训练下更稳。

### 7) GBPO-lite（`gbpo_clip`）

思路：按 advantage 符号做 sign-aware 裁剪：
- 正优势 token 用一组上下界
- 负优势 token 用另一组上下界

特点：对“该鼓励”和“该抑制”的更新分别控幅，通常能改善训练稳定性与鲁棒性。

### 8) Seq vs Token 聚合（`loss_aggregation`）

- `seq`：先按序列归一化再聚合，序列级信号更强，通常更稳。
- `token`：逐 token 聚合，粒度更细，可能更敏感，适合做细粒度 ablation。

### 9) Step-Progress GRPO（新增）

思路：在原有 GRPO advantage 上叠加“步骤进展奖励”：
1. 在 `<think>` 内按标点/显式 Step 词切步骤。  
2. 对每个步骤前缀做短 rollout，估计该前缀的终局成功率 `p_k`。  
3. 定义步骤即时奖励 `r_k = p_k - p_{k-1}`。  
4. 用 `(gamma, lambda)` 做步骤回报并映射到 token，得到 `token_step_adv`。  
5. 最终 `token_adv = token_seq_adv + alpha * token_step_adv`。

支持两种模式：
- `all`：所有样本都算 step-progress  
- `neg_only`：仅 all-negative group 启用（更省算力，也常更稳）

### 10) EI（Expert Iteration）筛题训练

思路：先让当前策略在训练集题目上多次采样，估计每题 pass-rate，再筛选题目做 SFT。  
支持：
- `static`：只在训练开始前筛一次题
- `dynamic`：每个 EI step 重新筛题

特点：通过“课程难度控制”把学习资源聚焦在当前策略可学区间，常用于提升数据效率。
