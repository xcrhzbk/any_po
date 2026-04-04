import argparse
import json
import os
import random
import re
import socket
from dataclasses import asdict, dataclass
from typing import Callable, List

import torch
import torch.nn as nn
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from unittest.mock import patch
from vllm import LLM, SamplingParams

from cs336_alignment.anypo import (
    compute_group_normalized_reward,
    grpo_microbatch_train_step,
    grpo_microbatch_train_step_seq_level_loss,
    grpo_microbatch_train_step_seq_level_loss_with_kl,
    grpo_microbatch_train_step_with_kl,
    masked_mean,
)
from cs336_alignment.drgrpo_grader_anypo import r1_zero_reward_fn
from cs336_alignment.math_baseline_anypo import run_vllm
from cs336_alignment.utils_anypo import get_response_log_probs, tokenize_prompt_and_output


@dataclass
class AnyPOConfig:
    model_path: str = "/home/bkzhu/storage/assignment5-alignment/data/a5-alignment/models/Qwen2.5-Math-1.5B"
    prompt_path: str = "/home/bkzhu/storage/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
    train_data_path: str = "/home/bkzhu/storage/assignment5-alignment/data/gsm8k/train.jsonl"
    test_data_path: str = "/home/bkzhu/storage/assignment5-alignment/data/gsm8k/test.jsonl"
    device_train: str = "cuda:0"
    device_vllm: str = "cuda:1"
    gpu_memory_utilization: float = 0.2
    seed: int = 69
    n_grpo_steps: int = 129
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 256
    group_size: int = 8
    train_batch_size: int = 256
    gradient_accumulation_steps: int = 32
    epochs_per_rollout_batch: int = 1
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4
    sampling_max_tokens: int = 512
    eval_max_tokens: int = 1024
    eval_freq: int = 8
    eval_num_samples: int = 0
    loss_type: str = "grpo_clip"
    loss_aggregation: str = "seq"  # seq | token
    use_std_normalization: bool = False
    use_kl_penalty: bool = False
    kl_beta: float = 0.02
    kl_estimator: str = "k3"  # k1 | k2 | k3
    kl_ref_mode: str = "old"  # currently only old is supported in this branch
    cliprange: float = 0.2
    cliprange_low: float = 0.2
    cliprange_high: float = 0.28
    gbpo_pos_cliprange_low: float = 0.2
    gbpo_pos_cliprange_high: float = 0.28
    gbpo_neg_cliprange_low: float = 0.05
    gbpo_neg_cliprange_high: float = 0.10
    enable_group_filter: bool = False
    filter_all_zero: bool = True
    filter_all_one: bool = True
    max_filter_resample_rounds: int = 8
    use_overlong_penalty: bool = False
    overlong_target_len: int = 256
    overlong_penalty_coef: float = 0.0
    project: str = "cs336-anypo"
    run_name: str = ""
    wandb_group: str = ""
    wandb_job_type: str = "train_anypo"
    wandb_tags: str = ""
    wandb_notes: str = ""
    save_dir: str = ""
    save_every: int = 0
    save_best: bool = True


ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")


def extract_reference_answer(answer: str) -> str:
    match = ANS_RE.search(answer)
    if match:
        return match.group(1).strip().replace(",", "")
    return "[invalid]"


def load_jsonl(file_path: str) -> list[dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def format_qa_prompt(data: list[dict], prompt_path: str) -> list[dict]:
    formatted_q = []
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()
    for d in data:
        pair = {"prompt": prompt.format(question=d["question"]), "answer": d["answer"]}
        formatted_q.append(pair)
    return formatted_q


def prepare_train_test(cfg: AnyPOConfig) -> tuple[list[dict], list[dict]]:
    train_data = load_jsonl(cfg.train_data_path)
    test_data = load_jsonl(cfg.test_data_path)
    train_data = format_qa_prompt(train_data, cfg.prompt_path)
    test_data = format_qa_prompt(test_data, cfg.prompt_path)
    return train_data, test_data


def init_vllm(model_id: str, device: str, gpu_memory_utilization: float = 0.85) -> LLM:
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams,
) -> dict[str, float]:
    responses = run_vllm(vllm_model, prompts, eval_sampling_params)
    allinfo_dict_list = []
    total_length = 0
    for response, answer in zip(responses, answers):
        extracted_answer = extract_reference_answer(answer)
        reward_dict = reward_fn(response, extracted_answer)
        allinfo_dict_list.append(reward_dict)
        total_length += len(response)

    overview = {"correct": 0, "format_wrong": 0, "answer_wrong": 0, "count": 0}
    for reward in allinfo_dict_list:
        overview["count"] += 1
        if reward["reward"] == 1:
            overview["correct"] += 1
        elif reward["format_reward"] == 1:
            overview["answer_wrong"] += 1
        else:
            overview["format_wrong"] += 1
    overview["avg_length"] = total_length / max(overview["count"], 1)
    return overview


def load_policy_into_vllm_instance(policy: torch.nn.Module, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def group_homogeneity_stats(raw_rewards: torch.Tensor, group_size: int) -> tuple[float, float]:
    rewards_per_group = raw_rewards.view(-1, group_size)
    all_zero = (rewards_per_group.sum(dim=-1) == 0).float().mean().item()
    all_one = (rewards_per_group.sum(dim=-1) == group_size).float().mean().item()
    return all_zero, all_one


def compute_reward_breakdown(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> dict[str, float]:
    reward_vals = []
    format_vals = []
    answer_vals = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(response, ground_truth)
        reward_vals.append(float(reward_dict["reward"]))
        format_vals.append(float(reward_dict["format_reward"]))
        answer_vals.append(float(reward_dict["answer_reward"]))

    n = max(len(reward_vals), 1)
    return {
        "reward_rate": sum(reward_vals) / n,
        "format_rate": sum(format_vals) / n,
        "answer_rate": sum(answer_vals) / n,
    }


def recompute_group_advantages_from_raw_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    advantage_eps: float,
    use_std_normalization: bool,
) -> torch.Tensor:
    rewards_per_group = raw_rewards.view(-1, group_size)
    advantages = rewards_per_group - rewards_per_group.mean(dim=-1, keepdim=True)
    if use_std_normalization:
        advantages = advantages / (rewards_per_group.std(dim=-1, keepdim=True) + advantage_eps)
    return advantages.reshape(-1)


def apply_overlong_penalty_to_rewards(
    raw_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    overlong_target_len: int,
    overlong_penalty_coef: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    response_len = response_mask.sum(dim=-1).to(dtype=torch.float32).cpu()
    target = float(max(overlong_target_len, 1))
    overlong_excess = torch.clamp(response_len - target, min=0.0)
    penalty = overlong_penalty_coef * (overlong_excess / target)
    return raw_rewards - penalty, penalty, overlong_excess


def sample_rollout_batch(
    train_data: list[dict],
    n_prompts: int,
) -> list[dict]:
    if len(train_data) >= n_prompts:
        return random.sample(train_data, n_prompts)
    return [random.choice(train_data) for _ in range(n_prompts)]


def collect_rollouts_with_optional_group_filter(
    cfg: AnyPOConfig,
    train_data: list[dict],
    vllm: LLM,
    n_prompts_per_rollout_batch: int,
) -> tuple[list[str], list[str], list[str], dict[str, float]]:
    prompts: list[str] = []
    responses: list[str] = []
    repeated_answers: list[str] = []
    accepted_groups = 0
    attempted_groups = 0
    rejected_groups = 0
    resample_rounds = 0

    max_rounds = cfg.max_filter_resample_rounds if cfg.enable_group_filter else 1
    while accepted_groups < n_prompts_per_rollout_batch and resample_rounds < max_rounds:
        resample_rounds += 1
        needed = n_prompts_per_rollout_batch - accepted_groups
        rollout_dataset = sample_rollout_batch(train_data, needed)
        rollout_prompts = [item["prompt"] for item in rollout_dataset]
        rollout_answers = [item["answer"] for item in rollout_dataset]
        sampling_params = SamplingParams(
            temperature=cfg.sampling_temperature,
            top_p=1.0,
            max_tokens=cfg.sampling_max_tokens,
            min_tokens=cfg.sampling_min_tokens,
            stop=["</answer>"],
            include_stop_str_in_output=True,
            n=cfg.group_size,
            seed=cfg.seed,
        )
        outputs = vllm.generate(rollout_prompts, sampling_params)
        attempted_groups += len(outputs)

        for output, answer in zip(outputs, rollout_answers):
            extracted_answer = extract_reference_answer(answer)
            group_responses = [r.text for r in output.outputs]
            group_rewards = [r1_zero_reward_fn(resp, extracted_answer)["reward"] for resp in group_responses]
            is_all_zero = sum(group_rewards) == 0
            is_all_one = sum(group_rewards) == cfg.group_size

            keep_group = True
            if cfg.enable_group_filter:
                if cfg.filter_all_zero and is_all_zero:
                    keep_group = False
                if cfg.filter_all_one and is_all_one:
                    keep_group = False

            if not keep_group:
                rejected_groups += 1
                continue

            accepted_groups += 1
            for group_response in group_responses:
                prompts.append(output.prompt)
                responses.append(group_response)
                repeated_answers.append(extracted_answer)

    filter_fallback_used = False
    if accepted_groups < n_prompts_per_rollout_batch:
        filter_fallback_used = True
        needed = n_prompts_per_rollout_batch - accepted_groups
        rollout_dataset = sample_rollout_batch(train_data, needed)
        rollout_prompts = [item["prompt"] for item in rollout_dataset]
        rollout_answers = [item["answer"] for item in rollout_dataset]
        sampling_params = SamplingParams(
            temperature=cfg.sampling_temperature,
            top_p=1.0,
            max_tokens=cfg.sampling_max_tokens,
            min_tokens=cfg.sampling_min_tokens,
            stop=["</answer>"],
            include_stop_str_in_output=True,
            n=cfg.group_size,
            seed=cfg.seed,
        )
        outputs = vllm.generate(rollout_prompts, sampling_params)
        attempted_groups += len(outputs)
        for output, answer in zip(outputs, rollout_answers):
            extracted_answer = extract_reference_answer(answer)
            accepted_groups += 1
            for group_response in output.outputs:
                prompts.append(output.prompt)
                responses.append(group_response.text)
                repeated_answers.append(extracted_answer)

    metadata = {
        "attempted_groups": float(attempted_groups),
        "accepted_groups": float(accepted_groups),
        "rejected_groups": float(rejected_groups),
        "resample_count": float(max(resample_rounds - 1, 0)),
        "valid_group_ratio": float(accepted_groups / max(attempted_groups, 1)),
        "filter_fallback_used": float(filter_fallback_used),
    }
    return prompts, responses, repeated_answers, metadata


def parse_wandb_tags(tags: str) -> list[str]:
    if not tags:
        return []
    return [tag.strip() for tag in tags.split(",") if tag.strip()]


def build_default_run_name(cfg: AnyPOConfig) -> str:
    return (
        f"anypo_{cfg.loss_type}"
        f"_agg-{cfg.loss_aggregation}"
        f"_std-{int(cfg.use_std_normalization)}"
        f"_g{cfg.group_size}"
        f"_seed{cfg.seed}"
    )


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer,
    cfg: AnyPOConfig,
    tag: str,
    step: int,
    eval_accuracy: float | None = None,
):
    if not cfg.save_dir:
        return
    ckpt_dir = os.path.join(cfg.save_dir, tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    meta = {"step": step, "eval_accuracy": eval_accuracy, "config": asdict(cfg)}
    with open(os.path.join(ckpt_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def train_anypo(cfg: AnyPOConfig):
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    assert cfg.loss_type in {
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_no_clip",
        "grpo_clip",
        "dapo_clip",
        "gbpo_clip",
    }
    assert cfg.loss_aggregation in {"seq", "token"}
    assert cfg.kl_estimator in {"k1", "k2", "k3"}
    assert cfg.kl_ref_mode in {"old"}
    assert cfg.train_batch_size % cfg.gradient_accumulation_steps == 0
    assert cfg.rollout_batch_size % cfg.group_size == 0
    assert cfg.train_batch_size >= cfg.group_size
    assert cfg.max_filter_resample_rounds >= 1

    micro_train_batch_size = cfg.train_batch_size // cfg.gradient_accumulation_steps
    n_prompts_per_rollout_batch = cfg.rollout_batch_size // cfg.group_size
    num_train_steps_per_epoch = cfg.rollout_batch_size // cfg.train_batch_size

    run_name = cfg.run_name or build_default_run_name(cfg)
    derived_config = {
        "micro_train_batch_size": micro_train_batch_size,
        "n_prompts_per_rollout_batch": n_prompts_per_rollout_batch,
        "num_train_steps_per_epoch": num_train_steps_per_epoch,
        "hostname": socket.gethostname(),
    }
    tags = parse_wandb_tags(cfg.wandb_tags)
    auto_tags = [
        f"loss:{cfg.loss_type}",
        f"agg:{cfg.loss_aggregation}",
        f"std:{int(cfg.use_std_normalization)}",
        f"group:{cfg.group_size}",
        f"group_filter:{int(cfg.enable_group_filter)}",
        f"overlong:{int(cfg.use_overlong_penalty)}",
        f"kl:{int(cfg.use_kl_penalty)}",
    ]
    tags = sorted(set(tags + auto_tags))
    wandb.init(
        project=cfg.project,
        name=run_name,
        group=cfg.wandb_group or None,
        job_type=cfg.wandb_job_type,
        notes=cfg.wandb_notes or None,
        tags=tags,
        config={**asdict(cfg), **derived_config},
    )
    wandb.define_metric("grpo_step")
    wandb.define_metric("train/update_step")
    wandb.define_metric("sampling/*", step_metric="grpo_step")
    wandb.define_metric("eval/*", step_metric="grpo_step")
    wandb.define_metric("train/*", step_metric="train/update_step")

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=cfg.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=cfg.device_train,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    vllm = init_vllm(cfg.model_path, cfg.device_vllm, cfg.gpu_memory_utilization)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    train_data, test_data = prepare_train_test(cfg)
    best_eval_acc = -1.0
    update_step = 0

    for grpo_step in range(cfg.n_grpo_steps):
        prompts, responses, repeated_answers, filter_meta = collect_rollouts_with_optional_group_filter(
            cfg,
            train_data,
            vllm,
            n_prompts_per_rollout_batch,
        )

        tokenization = tokenize_prompt_and_output(prompts, responses, tokenizer)
        input_ids = tokenization["input_ids"].to(cfg.device_train)
        labels = tokenization["labels"].to(cfg.device_train)
        response_mask = tokenization["response_mask"].to(cfg.device_train)

        advantages_train, raw_rewards_train, reward_meta = compute_group_normalized_reward(
            r1_zero_reward_fn,
            responses,
            repeated_answers,
            cfg.group_size,
            cfg.advantage_eps,
            cfg.use_std_normalization,
        )
        raw_rewards_binary = raw_rewards_train.clone()
        overlong_penalty = torch.zeros_like(raw_rewards_train, dtype=torch.float32)
        overlong_excess = torch.zeros_like(raw_rewards_train, dtype=torch.float32)
        if cfg.use_overlong_penalty and cfg.overlong_penalty_coef > 0:
            raw_rewards_train, overlong_penalty, overlong_excess = apply_overlong_penalty_to_rewards(
                raw_rewards_train,
                response_mask,
                cfg.overlong_target_len,
                cfg.overlong_penalty_coef,
            )
            advantages_train = recompute_group_advantages_from_raw_rewards(
                raw_rewards_train,
                cfg.group_size,
                cfg.advantage_eps,
                cfg.use_std_normalization,
            )

        all_negative_group_ratio, all_positive_group_ratio = group_homogeneity_stats(
            raw_rewards_binary, cfg.group_size
        )
        reward_breakdown = compute_reward_breakdown(r1_zero_reward_fn, responses, repeated_answers)
        response_lengths = torch.tensor([len(r) for r in responses], dtype=torch.float32)
        advantages_train = advantages_train.to(cfg.device_train)
        raw_rewards_train = raw_rewards_train.to(cfg.device_train)

        wandb.log(
            {
                "grpo_step": grpo_step,
                "sampling/avg_reward": reward_meta["mean"].item(),
                "sampling/avg_reward_shaped": raw_rewards_train.float().mean().item(),
                "sampling/reward_std": reward_meta["std"].item(),
                "sampling/reward_min": reward_meta["min"].item(),
                "sampling/reward_max": reward_meta["max"].item(),
                "sampling/reward_rate": reward_breakdown["reward_rate"],
                "sampling/format_rate": reward_breakdown["format_rate"],
                "sampling/answer_rate": reward_breakdown["answer_rate"],
                "sampling/all_negative_group_ratio": all_negative_group_ratio,
                "sampling/all_positive_group_ratio": all_positive_group_ratio,
                "sampling/response_len_mean": response_lengths.mean().item(),
                "sampling/response_len_std": response_lengths.std().item(),
                "sampling/response_len_min": response_lengths.min().item(),
                "sampling/response_len_max": response_lengths.max().item(),
                "sampling/adv_mean": advantages_train.float().mean().item(),
                "sampling/adv_std": advantages_train.float().std().item(),
                "sampling/adv_abs_mean": advantages_train.float().abs().mean().item(),
                "sampling/valid_group_ratio": filter_meta["valid_group_ratio"],
                "sampling/rejected_groups": filter_meta["rejected_groups"],
                "sampling/resample_count": filter_meta["resample_count"],
                "sampling/filter_fallback_used": filter_meta["filter_fallback_used"],
                "sampling/overlong_excess_mean": overlong_excess.mean().item(),
                "sampling/overlong_excess_max": overlong_excess.max().item(),
                "sampling/overlong_penalty_mean": overlong_penalty.mean().item(),
            },
        )

        with torch.no_grad():
            old_log_probs_train = []
            for train_step in range(num_train_steps_per_epoch):
                batch_start = train_step * cfg.train_batch_size
                for train_microstep in range(cfg.gradient_accumulation_steps):
                    micro_start = batch_start + train_microstep * micro_train_batch_size
                    micro_end = micro_start + micro_train_batch_size
                    log_probs_dict = get_response_log_probs(
                        model=model,
                        input_ids=input_ids[micro_start:micro_end],
                        labels=labels[micro_start:micro_end],
                        return_token_entropy=False,
                    )
                    old_log_probs_train.append(log_probs_dict["log_probs"])
            old_log_probs_train = torch.cat(old_log_probs_train, dim=0)

        for _ in range(cfg.epochs_per_rollout_batch):
            for train_step in range(num_train_steps_per_epoch):
                batch_start = train_step * cfg.train_batch_size
                batch_end = batch_start + cfg.train_batch_size
                batch_response_mask = response_mask[batch_start:batch_end]
                batch_mean_response_length = batch_response_mask.sum(dim=-1).mean(dtype=torch.float32)
                accumulated_token_entropy = 0.0
                accumulated_clip_fraction = 0.0
                accumulated_pg_loss = 0.0
                accumulated_kl_loss = 0.0
                accumulated_kl_mean = 0.0
                accumulated_clip_low_hit = 0.0
                accumulated_clip_high_hit = 0.0
                accumulated_gbpo_pos_hit = 0.0
                accumulated_gbpo_neg_hit = 0.0
                accumulated_ratio_extreme = 0.0

                for train_microstep in range(cfg.gradient_accumulation_steps):
                    micro_start = batch_start + train_microstep * micro_train_batch_size
                    micro_end = micro_start + micro_train_batch_size

                    raw_rewards = raw_rewards_train[micro_start:micro_end].unsqueeze(-1)
                    advantages = advantages_train[micro_start:micro_end].unsqueeze(-1)
                    old_log_probs = old_log_probs_train[micro_start:micro_end]
                    response_mask_micro_batch = response_mask[micro_start:micro_end]

                    log_probs_dict = get_response_log_probs(
                        model,
                        input_ids=input_ids[micro_start:micro_end],
                        labels=labels[micro_start:micro_end],
                        return_token_entropy=True,
                    )
                    policy_log_probs = log_probs_dict["log_probs"]
                    token_entropy = log_probs_dict["token_entropy"]
                    loss_kwargs = {
                        "raw_rewards": raw_rewards,
                        "advantages": advantages,
                        "old_log_probs": old_log_probs,
                        "cliprange": cfg.cliprange,
                        "cliprange_low": cfg.cliprange_low,
                        "cliprange_high": cfg.cliprange_high,
                        "gbpo_pos_cliprange_low": cfg.gbpo_pos_cliprange_low,
                        "gbpo_pos_cliprange_high": cfg.gbpo_pos_cliprange_high,
                        "gbpo_neg_cliprange_low": cfg.gbpo_neg_cliprange_low,
                        "gbpo_neg_cliprange_high": cfg.gbpo_neg_cliprange_high,
                    }
                    ref_log_probs = old_log_probs

                    if cfg.loss_aggregation == "seq":
                        if cfg.use_kl_penalty:
                            # KL reference uses old-policy log-probs on sampled actions.
                            loss, metadata = grpo_microbatch_train_step_seq_level_loss_with_kl(
                                policy_log_probs=policy_log_probs,
                                response_mask=response_mask_micro_batch,
                                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                                loss_type=cfg.loss_type,
                                ref_log_probs=ref_log_probs,
                                kl_beta=cfg.kl_beta,
                                kl_estimator=cfg.kl_estimator,
                                **loss_kwargs,
                            )
                        else:
                            loss, metadata = grpo_microbatch_train_step_seq_level_loss(
                                policy_log_probs=policy_log_probs,
                                response_mask=response_mask_micro_batch,
                                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                                loss_type=cfg.loss_type,
                                **loss_kwargs,
                            )
                    else:
                        if cfg.use_kl_penalty:
                            loss, metadata = grpo_microbatch_train_step_with_kl(
                                policy_log_probs=policy_log_probs,
                                response_mask=response_mask_micro_batch,
                                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                                loss_type=cfg.loss_type,
                                ref_log_probs=ref_log_probs,
                                kl_beta=cfg.kl_beta,
                                kl_estimator=cfg.kl_estimator,
                                **loss_kwargs,
                            )
                        else:
                            loss, metadata = grpo_microbatch_train_step(
                                policy_log_probs=policy_log_probs,
                                response_mask=response_mask_micro_batch,
                                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                                loss_type=cfg.loss_type,
                                **loss_kwargs,
                            )

                    accumulated_token_entropy += masked_mean(
                        token_entropy, response_mask_micro_batch, dim=None
                    ).item()
                    if "cliped" in metadata:
                        accumulated_clip_fraction += masked_mean(
                            metadata["cliped"], response_mask_micro_batch, dim=None
                        ).item()
                    if "clip_low_hit" in metadata:
                        accumulated_clip_low_hit += masked_mean(
                            metadata["clip_low_hit"], response_mask_micro_batch, dim=None
                        ).item()
                    if "clip_high_hit" in metadata:
                        accumulated_clip_high_hit += masked_mean(
                            metadata["clip_high_hit"], response_mask_micro_batch, dim=None
                        ).item()
                    if "gbpo_pos_clip_hit" in metadata:
                        accumulated_gbpo_pos_hit += masked_mean(
                            metadata["gbpo_pos_clip_hit"], response_mask_micro_batch, dim=None
                        ).item()
                    if "gbpo_neg_clip_hit" in metadata:
                        accumulated_gbpo_neg_hit += masked_mean(
                            metadata["gbpo_neg_clip_hit"], response_mask_micro_batch, dim=None
                        ).item()
                    if "ratio_extreme" in metadata:
                        accumulated_ratio_extreme += masked_mean(
                            metadata["ratio_extreme"], response_mask_micro_batch, dim=None
                        ).item()
                    if cfg.use_kl_penalty:
                        accumulated_pg_loss += masked_mean(
                            metadata["token_pg_loss"], response_mask_micro_batch, dim=None
                        ).item()
                        accumulated_kl_loss += masked_mean(
                            metadata["token_kl_loss"], response_mask_micro_batch, dim=None
                        ).item()
                        accumulated_kl_mean += masked_mean(
                            metadata["token_kl"], response_mask_micro_batch, dim=None
                        ).item()

                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                wandb.log(
                    {
                        "train/update_step": update_step,
                        "train/grpo_step": grpo_step,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/loss": loss.item() * cfg.gradient_accumulation_steps,
                        "train/avg_token_entropy": accumulated_token_entropy
                        / cfg.gradient_accumulation_steps,
                        "train/avg_clip_fraction": accumulated_clip_fraction
                        / cfg.gradient_accumulation_steps,
                        "train/grad_norm": grad_norm,
                        "train/mean_response_length": batch_mean_response_length,
                        "train/use_kl_penalty": float(cfg.use_kl_penalty),
                        "train/kl_beta": cfg.kl_beta if cfg.use_kl_penalty else 0.0,
                        "train/avg_pg_loss": accumulated_pg_loss / cfg.gradient_accumulation_steps
                        if cfg.use_kl_penalty
                        else 0.0,
                        "train/avg_kl_loss": accumulated_kl_loss / cfg.gradient_accumulation_steps
                        if cfg.use_kl_penalty
                        else 0.0,
                        "train/avg_kl": accumulated_kl_mean / cfg.gradient_accumulation_steps
                        if cfg.use_kl_penalty
                        else 0.0,
                        "train/clip_low_hit_rate": accumulated_clip_low_hit
                        / cfg.gradient_accumulation_steps,
                        "train/clip_high_hit_rate": accumulated_clip_high_hit
                        / cfg.gradient_accumulation_steps,
                        "train/gbpo_pos_clip_hit_rate": accumulated_gbpo_pos_hit
                        / cfg.gradient_accumulation_steps,
                        "train/gbpo_neg_clip_hit_rate": accumulated_gbpo_neg_hit
                        / cfg.gradient_accumulation_steps,
                        "train/ratio_extreme_rate": accumulated_ratio_extreme
                        / cfg.gradient_accumulation_steps,
                    },
                )
                update_step += 1

        load_policy_into_vllm_instance(model, vllm)

        do_eval = (grpo_step % cfg.eval_freq == 0) or (grpo_step == cfg.n_grpo_steps - 1)
        if do_eval:
            eval_data = test_data
            if cfg.eval_num_samples > 0:
                eval_data = test_data[: cfg.eval_num_samples]
            prompts = [data["prompt"] for data in eval_data]
            answers = [data["answer"] for data in eval_data]
            eval_sampling_params = SamplingParams(
                temperature=1.0,
                top_p=1.0,
                max_tokens=cfg.eval_max_tokens,
                stop=["</answer>"],
                include_stop_str_in_output=True,
            )
            overview = evaluate_vllm(vllm, r1_zero_reward_fn, prompts, answers, eval_sampling_params)
            eval_acc = overview["correct"] / max(overview["count"], 1)
            wandb.log(
                {
                    "grpo_step": grpo_step,
                    "eval/correct": overview["correct"],
                    "eval/correct format with wrong answer": overview["answer_wrong"],
                    "eval/wrong format": overview["format_wrong"],
                    "eval/accuracy": eval_acc,
                    "eval/avg_length": overview["avg_length"],
                },
            )

            if cfg.save_best and eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                save_checkpoint(model, tokenizer, cfg, "best", grpo_step, eval_acc)

        if cfg.save_every > 0 and (grpo_step + 1) % cfg.save_every == 0:
            save_checkpoint(model, tokenizer, cfg, f"step_{grpo_step + 1}", grpo_step + 1)

    save_checkpoint(model, tokenizer, cfg, "final", cfg.n_grpo_steps - 1, best_eval_acc)


def parse_args() -> AnyPOConfig:
    parser = argparse.ArgumentParser(description="Train AnyPO variants from one script.")
    parser.add_argument("--model_path", type=str, default=AnyPOConfig.model_path)
    parser.add_argument("--prompt_path", type=str, default=AnyPOConfig.prompt_path)
    parser.add_argument("--train_data_path", type=str, default=AnyPOConfig.train_data_path)
    parser.add_argument("--test_data_path", type=str, default=AnyPOConfig.test_data_path)
    parser.add_argument("--device_train", type=str, default=AnyPOConfig.device_train)
    parser.add_argument("--device_vllm", type=str, default=AnyPOConfig.device_vllm)
    parser.add_argument("--gpu_memory_utilization", type=float, default=AnyPOConfig.gpu_memory_utilization)
    parser.add_argument("--seed", type=int, default=AnyPOConfig.seed)
    parser.add_argument("--n_grpo_steps", type=int, default=AnyPOConfig.n_grpo_steps)
    parser.add_argument("--learning_rate", type=float, default=AnyPOConfig.learning_rate)
    parser.add_argument("--advantage_eps", type=float, default=AnyPOConfig.advantage_eps)
    parser.add_argument("--rollout_batch_size", type=int, default=AnyPOConfig.rollout_batch_size)
    parser.add_argument("--group_size", type=int, default=AnyPOConfig.group_size)
    parser.add_argument("--train_batch_size", type=int, default=AnyPOConfig.train_batch_size)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=AnyPOConfig.gradient_accumulation_steps,
    )
    parser.add_argument("--epochs_per_rollout_batch", type=int, default=AnyPOConfig.epochs_per_rollout_batch)
    parser.add_argument("--sampling_temperature", type=float, default=AnyPOConfig.sampling_temperature)
    parser.add_argument("--sampling_min_tokens", type=int, default=AnyPOConfig.sampling_min_tokens)
    parser.add_argument("--sampling_max_tokens", type=int, default=AnyPOConfig.sampling_max_tokens)
    parser.add_argument("--eval_max_tokens", type=int, default=AnyPOConfig.eval_max_tokens)
    parser.add_argument("--eval_freq", type=int, default=AnyPOConfig.eval_freq)
    parser.add_argument("--eval_num_samples", type=int, default=AnyPOConfig.eval_num_samples)
    parser.add_argument(
        "--loss_type",
        choices=[
            "no_baseline",
            "reinforce_with_baseline",
            "grpo_no_clip",
            "grpo_clip",
            "dapo_clip",
            "gbpo_clip",
        ],
        default=AnyPOConfig.loss_type,
    )
    parser.add_argument("--loss_aggregation", choices=["seq", "token"], default=AnyPOConfig.loss_aggregation)
    parser.add_argument(
        "--use_std_normalization",
        action=argparse.BooleanOptionalAction,
        default=AnyPOConfig.use_std_normalization,
    )
    parser.add_argument(
        "--use_kl_penalty",
        action=argparse.BooleanOptionalAction,
        default=AnyPOConfig.use_kl_penalty,
    )
    parser.add_argument("--kl_beta", type=float, default=AnyPOConfig.kl_beta)
    parser.add_argument(
        "--kl_estimator",
        choices=["k1", "k2", "k3"],
        default=AnyPOConfig.kl_estimator,
    )
    parser.add_argument(
        "--kl_ref_mode",
        choices=["old"],
        default=AnyPOConfig.kl_ref_mode,
    )
    parser.add_argument("--cliprange", type=float, default=AnyPOConfig.cliprange)
    parser.add_argument("--cliprange_low", type=float, default=AnyPOConfig.cliprange_low)
    parser.add_argument("--cliprange_high", type=float, default=AnyPOConfig.cliprange_high)
    parser.add_argument("--gbpo_pos_cliprange_low", type=float, default=AnyPOConfig.gbpo_pos_cliprange_low)
    parser.add_argument("--gbpo_pos_cliprange_high", type=float, default=AnyPOConfig.gbpo_pos_cliprange_high)
    parser.add_argument("--gbpo_neg_cliprange_low", type=float, default=AnyPOConfig.gbpo_neg_cliprange_low)
    parser.add_argument("--gbpo_neg_cliprange_high", type=float, default=AnyPOConfig.gbpo_neg_cliprange_high)
    parser.add_argument(
        "--enable_group_filter",
        action=argparse.BooleanOptionalAction,
        default=AnyPOConfig.enable_group_filter,
    )
    parser.add_argument(
        "--filter_all_zero",
        action=argparse.BooleanOptionalAction,
        default=AnyPOConfig.filter_all_zero,
    )
    parser.add_argument(
        "--filter_all_one",
        action=argparse.BooleanOptionalAction,
        default=AnyPOConfig.filter_all_one,
    )
    parser.add_argument(
        "--max_filter_resample_rounds",
        type=int,
        default=AnyPOConfig.max_filter_resample_rounds,
    )
    parser.add_argument(
        "--use_overlong_penalty",
        action=argparse.BooleanOptionalAction,
        default=AnyPOConfig.use_overlong_penalty,
    )
    parser.add_argument("--overlong_target_len", type=int, default=AnyPOConfig.overlong_target_len)
    parser.add_argument("--overlong_penalty_coef", type=float, default=AnyPOConfig.overlong_penalty_coef)
    parser.add_argument("--project", type=str, default=AnyPOConfig.project)
    parser.add_argument("--run_name", type=str, default=AnyPOConfig.run_name)
    parser.add_argument("--wandb_group", type=str, default=AnyPOConfig.wandb_group)
    parser.add_argument("--wandb_job_type", type=str, default=AnyPOConfig.wandb_job_type)
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default=AnyPOConfig.wandb_tags,
        help="Comma-separated tags, e.g. 'batch1,ablation,grpo_clip'",
    )
    parser.add_argument("--wandb_notes", type=str, default=AnyPOConfig.wandb_notes)
    parser.add_argument("--save_dir", type=str, default=AnyPOConfig.save_dir)
    parser.add_argument("--save_every", type=int, default=AnyPOConfig.save_every)
    parser.add_argument("--save_best", action=argparse.BooleanOptionalAction, default=AnyPOConfig.save_best)
    args = parser.parse_args()
    return AnyPOConfig(**vars(args))


if __name__ == "__main__":
    config = parse_args()
    train_anypo(config)
