import json
import os
import random
import re
import argparse
from argparse import ArgumentParser
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
from cs336_alignment.utils_anypo import (
    get_response_log_probs,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)


QWEN_MATH_BASE_PATH = "/home/bkzhu/storage/assignment5-alignment/data/a5-alignment/models/Qwen2.5-Math-1.5B"
PROMPT_PATH = "/home/bkzhu/storage/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
TEST_DATA_PATH = "/home/bkzhu/storage/assignment5-alignment/data/gsm8k/test.jsonl"
TRAIN_DATA_PATH = "/home/bkzhu/storage/assignment5-alignment/data/gsm8k/processed_train.jsonl"
OUTPUT_PATH = "/home/bkzhu/storage/assignment5-alignment/data/ei_zero"

# Dataset-agnostic: capture the full answer after "####" on that line.
ANS_RE = re.compile(r"####\s*([^\n\r]+)")
ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def to_float(val):
    if isinstance(val, torch.Tensor):
        return val.float().item()
    return float(val)


def load_jsonl(file_path: str) -> list[dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_reference_answer_gsm8k(answer: str) -> str:
    match = ANS_RE.search(answer)
    if match:
        return match.group(1).strip()
    return "[invalid]"


def extract_reference_answer_from_response(answer: str) -> str:
    match = ANSWER_TAG_RE.search(answer)
    if match:
        return match.group(1).strip()
    return "[invalid]"


def format_qa(data: list[dict]) -> list[dict]:
    formatted = []
    for d in data:
        formatted.append({"prompt": d["prompt"], "response": d["response"]})
    return formatted


def format_qa_prompt(data: list[dict], prompt_path: str) -> list[dict]:
    formatted_q = []
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()
    for d in data:
        formatted_q.append({"prompt": prompt.format(question=d["question"]), "answer": d["answer"]})
    return formatted_q


def prepare_train_test(train_data_path: str, test_data_path: str, prompt_path: str) -> tuple[list[dict], list[dict]]:
    train_data = format_qa(load_jsonl(train_data_path))
    test_data = format_qa_prompt(load_jsonl(test_data_path), prompt_path)
    return train_data, test_data


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85) -> LLM:
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
        extracted_answer = extract_reference_answer_gsm8k(answer)
        reward_dict = reward_fn(response, extracted_answer)
        overview["count"] += 1
        if reward_dict["reward"] == 1:
            overview["correct"] += 1
        elif reward_dict["format_reward"] == 1:
            overview["answer_wrong"] += 1
        else:
            overview["format_wrong"] += 1
    return overview


def get_ei_sft_batch(tokenized_train_data: dict[str, torch.Tensor], batch_size: int) -> dict[str, torch.Tensor]:
    total = len(tokenized_train_data["input_ids"])
    if total == 0:
        raise ValueError("Empty SFT dataset, cannot sample batch.")
    if total >= batch_size:
        batch_indices = random.sample(range(total), batch_size)
    else:
        batch_indices = [random.randrange(total) for _ in range(batch_size)]
    return {k: v[batch_indices] for k, v in tokenized_train_data.items()}


def chunk_list(data: list[dict], chunk_size: int) -> list[list[dict]]:
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


def compute_train_pass_rates(
    vllm: LLM,
    train_qa: list[dict],
    num_rollouts: int,
    batch_size: int,
    max_tokens: int,
    temperature: float,
) -> list[dict]:
    stats: list[dict] = []
    if len(train_qa) == 0:
        return stats

    for batch in chunk_list(train_qa, max(batch_size, 1)):
        prompts = [item["prompt"] for item in batch]
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
            min_tokens=4,
            stop=["</answer>"],
            include_stop_str_in_output=True,
            n=num_rollouts,
        )
        outputs = vllm.generate(prompts, sampling_params)
        for output, train_data in zip(outputs, batch):
            gt_answer = extract_reference_answer_from_response(train_data["response"])
            correct_count = 0
            for candidate in output.outputs:
                metrics = r1_zero_reward_fn(candidate.text.strip(), gt_answer)
                correct_count += int(metrics["reward"] > 0)
            pass_rate = correct_count / max(num_rollouts, 1)
            stats.append(
                {
                    "prompt": train_data["prompt"],
                    "response": train_data["response"],
                    "correct_count": correct_count,
                    "attempt_count": num_rollouts,
                    "pass_rate": pass_rate,
                }
            )
    return stats


def select_train_pool(
    train_qa: list[dict],
    train_stats: list[dict],
    threshold: float,
) -> tuple[list[dict], dict[str, float]]:
    if len(train_qa) != len(train_stats):
        raise ValueError("train_qa and train_stats must have same length")

    selected = []
    pass_rates = []
    for item, stat in zip(train_qa, train_stats):
        pass_rates.append(stat["pass_rate"])
        if stat["pass_rate"] >= threshold:
            selected.append(item)

    fallback_used = 0.0
    if len(selected) == 0 and len(train_qa) > 0:
        fallback_used = 1.0
        top_idx = max(range(len(train_stats)), key=lambda idx: train_stats[idx]["pass_rate"])
        selected = [train_qa[top_idx]]

    mean_pass = sum(pass_rates) / max(len(pass_rates), 1)
    return selected, {
        "selected_count": float(len(selected)),
        "selected_ratio": float(len(selected) / max(len(train_qa), 1)),
        "mean_pass_rate": float(mean_pass),
        "threshold": float(threshold),
        "selection_fallback_used": fallback_used,
    }


def sample_train_batch(train_qa: list[dict], batch_size: int) -> list[dict]:
    if len(train_qa) == 0:
        return []
    if len(train_qa) >= batch_size:
        batch_indices = random.sample(range(len(train_qa)), batch_size)
    else:
        batch_indices = [random.randrange(len(train_qa)) for _ in range(batch_size)]
    return [train_qa[i] for i in batch_indices]


def collect_expert_rollouts(
    vllm: LLM,
    selected_pool: list[dict],
    batch_size: int,
    ei_num_g: int,
    sampling_max_tokens: int = 512,
) -> tuple[list[dict], dict[str, float]]:
    batch = sample_train_batch(selected_pool, batch_size)
    if len(batch) == 0:
        return [], {
            "total": 0.0,
            "correct": 0.0,
            "format_wrong": 0.0,
            "format_correct_answer_wrong": 0.0,
            "rollout_accuracy": 0.0,
        }

    prompts = [item["prompt"] for item in batch]
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=sampling_max_tokens,
        min_tokens=4,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=ei_num_g,
    )
    outputs = vllm.generate(prompts, sampling_params)
    responses = [[o.text.strip() for o in output.outputs] for output in outputs]

    expert_roll = []
    overview = {"total": 0, "correct": 0, "format_wrong": 0, "format_correct_answer_wrong": 0}
    for response_group, train_data in zip(responses, batch):
        gt_answer = extract_reference_answer_from_response(train_data["response"])
        for rollout in response_group:
            overview["total"] += 1
            metrics = r1_zero_reward_fn(rollout, gt_answer)
            if metrics["reward"] > 0:
                overview["correct"] += 1
                expert_roll.append({"prompt": train_data["prompt"], "response": rollout})
            elif metrics["format_reward"] > 0:
                overview["format_correct_answer_wrong"] += 1
            else:
                overview["format_wrong"] += 1

    overview_float = {
        "total": float(overview["total"]),
        "correct": float(overview["correct"]),
        "format_wrong": float(overview["format_wrong"]),
        "format_correct_answer_wrong": float(overview["format_correct_answer_wrong"]),
        "rollout_accuracy": float(overview["correct"] / max(overview["total"], 1)),
    }
    return expert_roll, overview_float


def compute_step_threshold(args, ei_step: int) -> float:
    """Compute pass-rate threshold for the current EI step."""
    if not args.auto_threshold_decay:
        return float(args.train_pass_threshold)
    if args.n_ei_steps <= 1:
        return float(args.threshold_end)
    start = float(args.train_pass_threshold)
    end = float(args.threshold_end)
    progress = ei_step / float(max(args.n_ei_steps - 1, 1))
    return start + (end - start) * progress


def save_ei_checkpoint(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    output_path: str,
    ei_step: int,
    global_step: int,
    selection_meta: dict[str, float],
):
    ckpt_root = os.path.join(output_path, "checkpoints")
    ckpt_dir = os.path.join(ckpt_root, f"ei_step_{ei_step:04d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    state = {
        "ei_step": ei_step,
        "global_step": global_step,
        "selection_meta": selection_meta,
    }
    with open(os.path.join(ckpt_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    with open(os.path.join(ckpt_root, "latest_checkpoint.txt"), "w", encoding="utf-8") as f:
        f.write(ckpt_dir)


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


def load_resume_state(ckpt_dir: str) -> tuple[int, int]:
    state_path = os.path.join(ckpt_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        return 0, 0
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    return int(state.get("ei_step", -1)) + 1, int(state.get("global_step", 0))


def ei_sft(
    sft_data: list[dict[str, str]],
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    vllm: LLM,
    test_qa: list[dict[str, str]],
    epochs: int,
    micro_batch_size: int,
    n_grad_accum_steps: int,
    eval_steps: int,
    device_train: str,
    global_step: int = 0,
) -> int:
    if len(sft_data) == 0:
        return global_step

    tokenized_train_data = tokenize_prompt_and_output(
        prompt_strs=[data["prompt"] for data in sft_data],
        output_strs=[data["response"] for data in sft_data],
        tokenizer=tokenizer,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    amp_ctx = torch.amp.autocast(device_type=device_train, dtype=torch.bfloat16)

    n_sft_updates = max(len(sft_data) * epochs // max(n_grad_accum_steps * micro_batch_size, 1), 1)
    print(f"SFT updates in this EI step: {n_sft_updates}")

    for update_idx in range(n_sft_updates):
        for micro_idx in range(n_grad_accum_steps):
            with amp_ctx:
                train_batch = get_ei_sft_batch(tokenized_train_data, micro_batch_size)
                input_ids = train_batch["input_ids"].to(device_train)
                labels = train_batch["labels"].to(device_train)
                response_mask = train_batch["response_mask"].to(device_train)
                response_log_probs = get_response_log_probs(model, input_ids, labels, True)
                loss, _ = sft_microbatch_train_step(
                    response_log_probs["log_probs"], response_mask, n_grad_accum_steps
                )
                if micro_idx == n_grad_accum_steps - 1:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    entropy = response_log_probs["token_entropy"]

                    wandb.log(
                        {
                            "train/loss": to_float(loss),
                            "train/entropy": to_float(entropy.mean()),
                            "train/response_entropy": to_float(entropy[response_mask].mean()),
                            "train/prompt_entropy": to_float(entropy[~response_mask].mean()),
                            "train_step": global_step + 1,
                        }
                    )
                    global_step += 1

        if global_step % eval_steps == 0:
            load_policy_into_vllm_instance(model, vllm)
            sampling_params = SamplingParams(
                temperature=1.0,
                top_p=1.0,
                max_tokens=1024,
                min_tokens=4,
                stop=["</answer>"],
                include_stop_str_in_output=True,
            )
            test_prompt = [data["prompt"] for data in test_qa]
            test_answer = [data["answer"] for data in test_qa]
            overview = evaluate_vllm(vllm, r1_zero_reward_fn, test_prompt, test_answer, sampling_params)

            wandb.log(
                {
                    "eval/correct": overview["correct"],
                    "eval/correct format with wrong answer": overview["answer_wrong"],
                    "eval/wrong format": overview["format_wrong"],
                    "eval/accuracy": overview["correct"] / max(overview["count"], 1),
                    "eval_step": global_step + 1,
                }
            )

            print(f"Eval at update {update_idx}/{n_sft_updates}:")
            print(f"  correct={overview['correct']}")
            print(f"  accuracy={overview['correct'] / max(overview['count'], 1) * 100:.2f}%")
            print(f"  format_correct_answer_wrong={overview['answer_wrong']}")
            print(f"  format_wrong={overview['format_wrong']}")
    return global_step


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    wandb.init(
        project=args.project,
        name=args.run_name
        or f"ei_{args.selection_mode}_thr{args.train_pass_threshold}_steps{args.n_ei_steps}",
        config=vars(args),
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("ei_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    wandb.define_metric("selection/*", step_metric="ei_step")
    wandb.define_metric("rollout/*", step_metric="ei_step")

    resume_ckpt = resolve_resume_checkpoint(args.output_path, args.resume_from, args.auto_resume)
    if resume_ckpt:
        model_source = resume_ckpt
        start_ei_step, global_step = load_resume_state(resume_ckpt)
        print(f"Resuming from {resume_ckpt}, start_ei_step={start_ei_step}, global_step={global_step}")
    else:
        model_source = args.model_path
        start_ei_step = 0
        global_step = 0

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_source,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=args.device_train,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    vllm = init_vllm(args.model_path, args.device_vllm, args.seed, gpu_memory_utilization=args.gpu_memory_utilization)
    load_policy_into_vllm_instance(model, vllm)

    train_qa, test_qa = prepare_train_test(args.train_data_path, args.test_data_path, args.prompt_path)
    static_selected_pool: list[dict] | None = None
    static_selection_meta: dict[str, float] | None = None

    if args.selection_mode == "static":
        if args.auto_threshold_decay:
            raise ValueError("selection_mode=static does not support --auto_threshold_decay.")
        current_threshold = float(args.train_pass_threshold)
        selection_rollouts = args.selection_num_g if args.selection_num_g > 0 else args.ei_num_g
        train_stats = compute_train_pass_rates(
            vllm,
            train_qa,
            selection_rollouts,
            args.selection_batch_size,
            args.selection_max_tokens,
            args.selection_temperature,
        )
        static_selected_pool, static_selection_meta = select_train_pool(
            train_qa, train_stats, current_threshold
        )
        wandb.log(
            {
                "ei_step": start_ei_step,
                "selection/selected_count": static_selection_meta["selected_count"],
                "selection/selected_ratio": static_selection_meta["selected_ratio"],
                "selection/mean_pass_rate": static_selection_meta["mean_pass_rate"],
                "selection/threshold": static_selection_meta["threshold"],
                "selection/fallback_used": static_selection_meta["selection_fallback_used"],
            }
        )

    for ei_step in range(start_ei_step, args.n_ei_steps):
        print("------------------------------")
        print(f"Expert iteration step: {ei_step}")
        print(f"Global step now: {global_step}")
        current_threshold = compute_step_threshold(args, ei_step)
        print(f"Current pass-rate threshold: {current_threshold:.4f}")

        if args.selection_mode == "dynamic":
            selection_rollouts = args.selection_num_g if args.selection_num_g > 0 else args.ei_num_g
            train_stats = compute_train_pass_rates(
                vllm,
                train_qa,
                selection_rollouts,
                args.selection_batch_size,
                args.selection_max_tokens,
                args.selection_temperature,
            )
            selected_pool, selection_meta = select_train_pool(train_qa, train_stats, current_threshold)
        elif args.selection_mode == "static":
            selected_pool = static_selected_pool if static_selected_pool is not None else train_qa
            selection_meta = static_selection_meta if static_selection_meta is not None else {
                "selected_count": float(len(selected_pool)),
                "selected_ratio": float(len(selected_pool) / max(len(train_qa), 1)),
                "mean_pass_rate": 0.0,
                "threshold": current_threshold,
                "selection_fallback_used": 0.0,
            }
        else:
            selected_pool = train_qa
            selection_meta = {
                "selected_count": float(len(selected_pool)),
                "selected_ratio": 1.0,
                "mean_pass_rate": 0.0,
                "threshold": current_threshold,
                "selection_fallback_used": 0.0,
            }

        wandb.log(
            {
                "ei_step": ei_step,
                "selection/selected_count": selection_meta["selected_count"],
                "selection/selected_ratio": selection_meta["selected_ratio"],
                "selection/mean_pass_rate": selection_meta["mean_pass_rate"],
                "selection/threshold": selection_meta["threshold"],
                "selection/fallback_used": selection_meta["selection_fallback_used"],
            }
        )

        sft_data, rollout_meta = collect_expert_rollouts(
            vllm,
            selected_pool,
            args.batch_size,
            args.ei_num_g,
            sampling_max_tokens=args.rollout_max_tokens,
        )
        wandb.log(
            {
                "ei_step": ei_step,
                "rollout/total": rollout_meta["total"],
                "rollout/correct": rollout_meta["correct"],
                "rollout/format_wrong": rollout_meta["format_wrong"],
                "rollout/format_correct_answer_wrong": rollout_meta["format_correct_answer_wrong"],
                "rollout/accuracy": rollout_meta["rollout_accuracy"],
                "rollout/sft_data_size": float(len(sft_data)),
            }
        )
        print(
            "rollout summary: "
            f"acc={rollout_meta['rollout_accuracy']*100:.2f}% "
            f"correct={int(rollout_meta['correct'])} total={int(rollout_meta['total'])} "
            f"sft_data_size={len(sft_data)}"
        )

        global_step = ei_sft(
            sft_data=sft_data,
            model=model,
            tokenizer=tokenizer,
            vllm=vllm,
            test_qa=test_qa,
            epochs=args.epochs,
            micro_batch_size=args.micro_batch_size,
            n_grad_accum_steps=args.n_grad_accum_steps,
            eval_steps=args.eval_steps,
            device_train=args.device_train,
            global_step=global_step,
        )
        load_policy_into_vllm_instance(model, vllm)
        save_ei_checkpoint(model, tokenizer, args.output_path, ei_step, global_step, selection_meta)

    final_dir = os.path.join(args.output_path, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Training finished. Final checkpoint saved to {final_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--project", type=str, default="cs336-ei-sft")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--model_path", type=str, default=QWEN_MATH_BASE_PATH)
    parser.add_argument("--prompt_path", type=str, default=PROMPT_PATH)
    parser.add_argument("--train_data_path", type=str, default=TRAIN_DATA_PATH)
    parser.add_argument("--test_data_path", type=str, default=TEST_DATA_PATH)
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--device_train", type=str, default="cuda:1")
    parser.add_argument("--device_vllm", type=str, default="cuda:2")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.3)

    parser.add_argument("--n_ei_steps", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=7000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--ei_num_g", type=int, default=2)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--n_grad_accum_steps", type=int, default=8)
    parser.add_argument("--eval_steps", type=int, default=16)
    parser.add_argument("--rollout_max_tokens", type=int, default=512)

    parser.add_argument("--selection_mode", choices=["none", "static", "dynamic"], default="none")
    parser.add_argument("--train_pass_threshold", type=float, default=0.5)
    parser.add_argument(
        "--auto_threshold_decay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Linearly decay threshold from train_pass_threshold to threshold_end across EI steps.",
    )
    parser.add_argument("--threshold_end", type=float, default=0.0)
    parser.add_argument("--selection_num_g", type=int, default=0)
    parser.add_argument("--selection_batch_size", type=int, default=256)
    parser.add_argument("--selection_max_tokens", type=int, default=512)
    parser.add_argument("--selection_temperature", type=float, default=1.0)

    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--auto_resume", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    main(args)
