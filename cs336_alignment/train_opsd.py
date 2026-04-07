import argparse
import json
import os
import random
import re
import socket
from dataclasses import asdict, dataclass
from typing import Callable, List
from unittest.mock import patch

import torch
import torch.nn as nn
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from cs336_alignment.drgrpo_grader_anypo import r1_zero_reward_fn
from cs336_alignment.math_baseline_anypo import run_vllm
from cs336_alignment.opsd import compute_masked_token_kl_loss
from cs336_alignment.utils_anypo import tokenize_prompt_and_output


@dataclass
class OPSDConfig:
    model_path: str = "/home/bkzhu/storage/assignment5-alignment/data/a5-alignment/models/Qwen2.5-Math-1.5B"
    prompt_path: str = "/home/bkzhu/storage/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
    train_data_path: str = "/home/bkzhu/storage/assignment5-alignment/data/gsm8k/train.jsonl"
    test_data_path: str = "/home/bkzhu/storage/assignment5-alignment/data/gsm8k/test.jsonl"
    output_path: str = "/home/bkzhu/storage/assignment5-alignment/data/opsd_ckpts"
    device_train: str = "cuda:0"
    device_vllm: str = "cuda:1"
    gpu_memory_utilization: float = 0.3
    seed: int = 69
    project: str = "cs336-opsd"
    run_name: str = ""
    wandb_group: str = ""
    wandb_job_type: str = "train_opsd"
    wandb_tags: str = ""
    wandb_notes: str = ""
    n_opsd_steps: int = 20
    prompt_batch_size: int = 256
    epochs_per_step: int = 2
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-5
    kl_coef: float = 1.0
    teacher_use_ground_truth: bool = True
    teacher_hint_template: str = (
        "\n\n[Reference Answer]\n{answer}\n"
        "Use the reference answer to guide reasoning, then output <think>...</think> <answer>...</answer>."
    )
    rollout_temperature: float = 1.0
    rollout_min_tokens: int = 4
    rollout_max_tokens: int = 512
    eval_steps: int = 16
    eval_max_tokens: int = 1024
    eval_num_samples: int = 0
    save_best: bool = True
    save_every: int = 0
    resume_from: str = ""
    auto_resume: bool = True


ANS_RE = re.compile(r"####\s*([^\n\r]+)")


def parse_wandb_tags(tags: str) -> list[str]:
    if not tags:
        return []
    return [t.strip() for t in tags.split(",") if t.strip()]


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_reference_answer(answer: str) -> str:
    match = ANS_RE.search(answer)
    if match:
        return match.group(1).strip()
    return "[invalid]"


def format_qa_prompt(data: list[dict], prompt_path: str) -> list[dict]:
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()
    formatted = []
    for d in data:
        formatted.append(
            {
                "prompt": prompt.format(question=d["question"]),
                "answer": extract_reference_answer(d["answer"]),
                "raw_answer": d["answer"],
            }
        )
    return formatted


def prepare_train_test(cfg: OPSDConfig) -> tuple[list[dict], list[dict]]:
    train_data = format_qa_prompt(load_jsonl(cfg.train_data_path), cfg.prompt_path)
    test_data = format_qa_prompt(load_jsonl(cfg.test_data_path), cfg.prompt_path)
    return train_data, test_data


def sample_train_batch(train_data: list[dict], batch_size: int) -> list[dict]:
    if len(train_data) == 0:
        return []
    if len(train_data) >= batch_size:
        idx = random.sample(range(len(train_data)), batch_size)
    else:
        idx = [random.randrange(len(train_data)) for _ in range(batch_size)]
    return [train_data[i] for i in idx]


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float) -> LLM:
    vllm_set_random_seed(seed)
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


def load_policy_into_vllm_instance(policy: torch.nn.Module, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams,
) -> dict[str, float]:
    responses = run_vllm(vllm_model, prompts, eval_sampling_params)
    overview = {"correct": 0, "format_wrong": 0, "answer_wrong": 0, "count": 0}
    for response, answer in zip(responses, answers):
        reward_dict = reward_fn(response, answer)
        overview["count"] += 1
        if reward_dict["reward"] == 1:
            overview["correct"] += 1
        elif reward_dict["format_reward"] == 1:
            overview["answer_wrong"] += 1
        else:
            overview["format_wrong"] += 1
    return overview


def build_teacher_prompts(cfg: OPSDConfig, prompts: list[str], answers: list[str]) -> list[str]:
    if not cfg.teacher_use_ground_truth:
        return prompts
    teacher_prompts = []
    for prompt, answer in zip(prompts, answers):
        teacher_prompts.append(prompt + cfg.teacher_hint_template.format(answer=answer))
    return teacher_prompts


def rollout_and_stats(
    cfg: OPSDConfig,
    vllm: LLM,
    batch: list[dict],
) -> tuple[list[str], list[str], list[str], dict[str, float]]:
    prompts = [x["prompt"] for x in batch]
    answers = [x["answer"] for x in batch]
    sampling_params = SamplingParams(
        temperature=cfg.rollout_temperature,
        top_p=1.0,
        max_tokens=cfg.rollout_max_tokens,
        min_tokens=cfg.rollout_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=1,
        seed=cfg.seed,
    )
    outputs = vllm.generate(prompts, sampling_params)
    responses = [output.outputs[0].text.strip() for output in outputs]

    reward_vals = []
    format_vals = []
    answer_vals = []
    lengths = []
    for response, answer in zip(responses, answers):
        reward = r1_zero_reward_fn(response, answer)
        reward_vals.append(float(reward["reward"]))
        format_vals.append(float(reward["format_reward"]))
        answer_vals.append(float(reward["answer_reward"]))
        lengths.append(float(len(response)))

    n = max(len(responses), 1)
    meta = {
        "reward_rate": sum(reward_vals) / n,
        "format_rate": sum(format_vals) / n,
        "answer_rate": sum(answer_vals) / n,
        "response_len_mean": sum(lengths) / n,
        "response_len_max": max(lengths) if lengths else 0.0,
        "response_len_min": min(lengths) if lengths else 0.0,
    }
    return prompts, answers, responses, meta


def save_step_checkpoint(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    cfg: OPSDConfig,
    opsd_step: int,
    global_update_step: int,
    best_eval_acc: float,
):
    ckpt_root = os.path.join(cfg.output_path, "checkpoints")
    ckpt_dir = os.path.join(ckpt_root, f"opsd_step_{opsd_step:04d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    state = {
        "opsd_step": opsd_step,
        "global_update_step": global_update_step,
        "best_eval_acc": best_eval_acc,
    }
    with open(os.path.join(ckpt_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    with open(os.path.join(ckpt_root, "latest_checkpoint.txt"), "w", encoding="utf-8") as f:
        f.write(ckpt_dir)


def save_named_checkpoint(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    cfg: OPSDConfig,
    tag: str,
    opsd_step: int,
    global_update_step: int,
    eval_acc: float,
):
    ckpt_dir = os.path.join(cfg.output_path, tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    meta = {
        "opsd_step": opsd_step,
        "global_update_step": global_update_step,
        "eval_acc": eval_acc,
        "config": asdict(cfg),
    }
    with open(os.path.join(ckpt_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def resolve_resume_checkpoint(output_path: str, resume_from: str, auto_resume: bool) -> str:
    if resume_from:
        return resume_from
    if not auto_resume:
        return ""
    latest_path = os.path.join(output_path, "checkpoints", "latest_checkpoint.txt")
    if not os.path.exists(latest_path):
        return ""
    with open(latest_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_resume_state(ckpt_dir: str) -> tuple[int, int, float]:
    state_path = os.path.join(ckpt_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        return 0, 0, -1.0
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    return (
        int(state.get("opsd_step", -1)) + 1,
        int(state.get("global_update_step", 0)),
        float(state.get("best_eval_acc", -1.0)),
    )


def train_opsd(cfg: OPSDConfig):
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    assert cfg.n_opsd_steps >= 1
    assert cfg.prompt_batch_size >= 1
    assert cfg.epochs_per_step >= 1
    assert cfg.micro_batch_size >= 1
    assert cfg.gradient_accumulation_steps >= 1
    assert cfg.eval_steps >= 1

    resume_ckpt = resolve_resume_checkpoint(cfg.output_path, cfg.resume_from, cfg.auto_resume)
    if resume_ckpt:
        model_source = resume_ckpt
        start_opsd_step, global_update_step, best_eval_acc = load_resume_state(resume_ckpt)
    else:
        model_source = cfg.model_path
        start_opsd_step = 0
        global_update_step = 0
        best_eval_acc = -1.0

    run_name = cfg.run_name or f"opsd_gt{int(cfg.teacher_use_ground_truth)}_k{cfg.kl_coef}"
    tags = parse_wandb_tags(cfg.wandb_tags)
    auto_tags = [
        "opsd",
        f"gt:{int(cfg.teacher_use_ground_truth)}",
        f"k:{cfg.kl_coef}",
    ]
    tags = sorted(set(tags + auto_tags))
    wandb.init(
        project=cfg.project,
        name=run_name,
        group=cfg.wandb_group or None,
        job_type=cfg.wandb_job_type,
        notes=cfg.wandb_notes or None,
        tags=tags,
        config={**asdict(cfg), "hostname": socket.gethostname()},
    )
    wandb.define_metric("opsd_step")
    wandb.define_metric("train/update_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("sampling/*", step_metric="opsd_step")
    wandb.define_metric("train/*", step_metric="train/update_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_source,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=cfg.device_train,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    vllm = init_vllm(cfg.model_path, cfg.device_vllm, cfg.seed, cfg.gpu_memory_utilization)
    load_policy_into_vllm_instance(model, vllm)

    train_data, test_data = prepare_train_test(cfg)

    for opsd_step in range(start_opsd_step, cfg.n_opsd_steps):
        batch = sample_train_batch(train_data, cfg.prompt_batch_size)
        if len(batch) == 0:
            break

        prompts, answers, responses, sample_meta = rollout_and_stats(cfg, vllm, batch)
        teacher_prompts = build_teacher_prompts(cfg, prompts, answers)

        student_tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
        teacher_tokenized = tokenize_prompt_and_output(teacher_prompts, responses, tokenizer)

        student_input_ids = student_tokenized["input_ids"].to(cfg.device_train)
        student_response_mask = student_tokenized["response_mask"].to(cfg.device_train)
        teacher_input_ids = teacher_tokenized["input_ids"].to(cfg.device_train)
        teacher_response_mask = teacher_tokenized["response_mask"].to(cfg.device_train)

        batch_size = student_input_ids.shape[0]
        updates_per_epoch = max(
            batch_size // max(cfg.micro_batch_size * cfg.gradient_accumulation_steps, 1),
            1,
        )

        wandb.log(
            {
                "opsd_step": opsd_step,
                "sampling/reward_rate": sample_meta["reward_rate"],
                "sampling/format_rate": sample_meta["format_rate"],
                "sampling/answer_rate": sample_meta["answer_rate"],
                "sampling/response_len_mean": sample_meta["response_len_mean"],
                "sampling/response_len_min": sample_meta["response_len_min"],
                "sampling/response_len_max": sample_meta["response_len_max"],
                "sampling/teacher_use_ground_truth": float(cfg.teacher_use_ground_truth),
            }
        )

        for _ in range(cfg.epochs_per_step):
            for _update in range(updates_per_epoch):
                accumulated_kl = 0.0
                accumulated_aligned_tokens = 0.0
                accumulated_length_mismatch = 0.0
                for _micro in range(cfg.gradient_accumulation_steps):
                    idx = torch.randint(0, batch_size, (cfg.micro_batch_size,), device=cfg.device_train)
                    s_in = student_input_ids[idx]
                    t_in = teacher_input_ids[idx]
                    s_mask = student_response_mask[idx]
                    t_mask = teacher_response_mask[idx]

                    student_logits = model(s_in).logits
                    with torch.no_grad():
                        teacher_logits = model(t_in).logits

                    raw_loss, meta = compute_masked_token_kl_loss(
                        student_logits=student_logits,
                        teacher_logits=teacher_logits,
                        student_response_mask=s_mask,
                        teacher_response_mask=t_mask,
                        kl_coef=cfg.kl_coef,
                    )
                    loss = raw_loss / cfg.gradient_accumulation_steps
                    loss.backward()
                    accumulated_kl += meta["kl_mean"].item()
                    accumulated_aligned_tokens += meta["aligned_tokens"].item()
                    accumulated_length_mismatch += meta["length_mismatch_count"].item()

                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_update_step += 1

                wandb.log(
                    {
                        "train/update_step": global_update_step,
                        "train/kl_loss": accumulated_kl / cfg.gradient_accumulation_steps,
                        "train/aligned_tokens": accumulated_aligned_tokens / cfg.gradient_accumulation_steps,
                        "train/length_mismatch_count": accumulated_length_mismatch / cfg.gradient_accumulation_steps,
                        "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                        "train/lr": optimizer.param_groups[0]["lr"],
                    }
                )

                if global_update_step % cfg.eval_steps == 0:
                    load_policy_into_vllm_instance(model, vllm)
                    eval_data = test_data if cfg.eval_num_samples <= 0 else test_data[: cfg.eval_num_samples]
                    eval_prompts = [x["prompt"] for x in eval_data]
                    eval_answers = [x["answer"] for x in eval_data]
                    eval_sampling_params = SamplingParams(
                        temperature=1.0,
                        top_p=1.0,
                        max_tokens=cfg.eval_max_tokens,
                        min_tokens=cfg.rollout_min_tokens,
                        stop=["</answer>"],
                        include_stop_str_in_output=True,
                    )
                    overview = evaluate_vllm(
                        vllm_model=vllm,
                        reward_fn=r1_zero_reward_fn,
                        prompts=eval_prompts,
                        answers=eval_answers,
                        eval_sampling_params=eval_sampling_params,
                    )
                    eval_acc = overview["correct"] / max(overview["count"], 1)
                    wandb.log(
                        {
                            "eval_step": global_update_step,
                            "eval/correct": overview["correct"],
                            "eval/correct_format_wrong_answer": overview["answer_wrong"],
                            "eval/wrong_format": overview["format_wrong"],
                            "eval/accuracy": eval_acc,
                        }
                    )
                    if cfg.save_best and eval_acc > best_eval_acc:
                        best_eval_acc = eval_acc
                        save_named_checkpoint(
                            model,
                            tokenizer,
                            cfg,
                            "best",
                            opsd_step=opsd_step,
                            global_update_step=global_update_step,
                            eval_acc=eval_acc,
                        )

        load_policy_into_vllm_instance(model, vllm)
        save_step_checkpoint(
            model,
            tokenizer,
            cfg,
            opsd_step=opsd_step,
            global_update_step=global_update_step,
            best_eval_acc=best_eval_acc,
        )
        if cfg.save_every > 0 and (opsd_step + 1) % cfg.save_every == 0:
            save_named_checkpoint(
                model,
                tokenizer,
                cfg,
                f"step_{opsd_step + 1}",
                opsd_step=opsd_step,
                global_update_step=global_update_step,
                eval_acc=best_eval_acc,
            )

    save_named_checkpoint(
        model,
        tokenizer,
        cfg,
        "final",
        opsd_step=max(cfg.n_opsd_steps - 1, 0),
        global_update_step=global_update_step,
        eval_acc=best_eval_acc,
    )


def parse_args() -> OPSDConfig:
    parser = argparse.ArgumentParser(description="Train OPSD (on-policy self-distillation).")
    parser.add_argument("--model_path", type=str, default=OPSDConfig.model_path)
    parser.add_argument("--prompt_path", type=str, default=OPSDConfig.prompt_path)
    parser.add_argument("--train_data_path", type=str, default=OPSDConfig.train_data_path)
    parser.add_argument("--test_data_path", type=str, default=OPSDConfig.test_data_path)
    parser.add_argument("--output_path", type=str, default=OPSDConfig.output_path)
    parser.add_argument("--device_train", type=str, default=OPSDConfig.device_train)
    parser.add_argument("--device_vllm", type=str, default=OPSDConfig.device_vllm)
    parser.add_argument("--gpu_memory_utilization", type=float, default=OPSDConfig.gpu_memory_utilization)
    parser.add_argument("--seed", type=int, default=OPSDConfig.seed)
    parser.add_argument("--project", type=str, default=OPSDConfig.project)
    parser.add_argument("--run_name", type=str, default=OPSDConfig.run_name)
    parser.add_argument("--wandb_group", type=str, default=OPSDConfig.wandb_group)
    parser.add_argument("--wandb_job_type", type=str, default=OPSDConfig.wandb_job_type)
    parser.add_argument("--wandb_tags", type=str, default=OPSDConfig.wandb_tags)
    parser.add_argument("--wandb_notes", type=str, default=OPSDConfig.wandb_notes)
    parser.add_argument("--n_opsd_steps", type=int, default=OPSDConfig.n_opsd_steps)
    parser.add_argument("--prompt_batch_size", type=int, default=OPSDConfig.prompt_batch_size)
    parser.add_argument("--epochs_per_step", type=int, default=OPSDConfig.epochs_per_step)
    parser.add_argument("--micro_batch_size", type=int, default=OPSDConfig.micro_batch_size)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=OPSDConfig.gradient_accumulation_steps)
    parser.add_argument("--learning_rate", type=float, default=OPSDConfig.learning_rate)
    parser.add_argument("--kl_coef", type=float, default=OPSDConfig.kl_coef)
    parser.add_argument(
        "--teacher_use_ground_truth",
        action=argparse.BooleanOptionalAction,
        default=OPSDConfig.teacher_use_ground_truth,
    )
    parser.add_argument("--teacher_hint_template", type=str, default=OPSDConfig.teacher_hint_template)
    parser.add_argument("--rollout_temperature", type=float, default=OPSDConfig.rollout_temperature)
    parser.add_argument("--rollout_min_tokens", type=int, default=OPSDConfig.rollout_min_tokens)
    parser.add_argument("--rollout_max_tokens", type=int, default=OPSDConfig.rollout_max_tokens)
    parser.add_argument("--eval_steps", type=int, default=OPSDConfig.eval_steps)
    parser.add_argument("--eval_max_tokens", type=int, default=OPSDConfig.eval_max_tokens)
    parser.add_argument("--eval_num_samples", type=int, default=OPSDConfig.eval_num_samples)
    parser.add_argument("--save_best", action=argparse.BooleanOptionalAction, default=OPSDConfig.save_best)
    parser.add_argument("--save_every", type=int, default=OPSDConfig.save_every)
    parser.add_argument("--resume_from", type=str, default=OPSDConfig.resume_from)
    parser.add_argument("--auto_resume", action=argparse.BooleanOptionalAction, default=OPSDConfig.auto_resume)
    args = parser.parse_args()
    return OPSDConfig(**vars(args))


if __name__ == "__main__":
    config = parse_args()
    train_opsd(config)
