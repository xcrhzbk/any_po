import torch
from einops import repeat
from typing import Literal
from cs336_alignment.utils_anypo import masked_normalize


LossType = Literal[
    "no_baseline",
    "reinforce_with_baseline",
    "grpo_clip",
    "grpo_no_clip",
    "dapo_clip",
    "gbpo_clip",
]


def compute_group_normalized_reward(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalized_by_std
):
    """reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against
the ground truths, producing a dict with keys "reward", "format_reward", and
"answer_reward".
"""
    raw_rewards = []
    for rollout_response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(rollout_response, ground_truth)["reward"]
        raw_rewards.append(reward_dict)
    raw_rewards = torch.tensor(raw_rewards) # (prompts * group_size, )
    reward_per_group = raw_rewards.view(-1, group_size) # (prompts, group_size)
    mean_reward_per_group = reward_per_group.mean(dim=-1, keepdim=True) # (prompts, 1)
    advantages = reward_per_group - mean_reward_per_group # (prompts, group_size)
    if normalized_by_std:
        std_reward_per_group = torch.std(reward_per_group, dim=-1, keepdim=True)# (prompts, 1)
        advantages = advantages / (std_reward_per_group + advantage_eps)
    advantages = repeat(advantages, "b g -> (b g)") # (prompts * group_size, )

    metadata = {
        "mean": torch.mean(raw_rewards),
        "std": torch.std(raw_rewards),
        "max": torch.max(raw_rewards),
        "min": torch.min(raw_rewards),
    }
    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,    # batch_size, 1
    policy_log_probs: torch.Tensor       # batch_size,  seq_len
) -> torch.Tensor: 
    _, seq_len = policy_log_probs.shape
    raw_rewards_or_advantages = repeat(
        raw_rewards_or_advantages,
        "b 1 -> b s",
        s=seq_len
    ) # (batch_size, seq_len)
    loss = -raw_rewards_or_advantages * policy_log_probs # (batch_size, seq_len)
    return loss


def _expand_advantages(advantages: torch.Tensor, seq_len: int) -> torch.Tensor:
    return repeat(advantages, "b 1 -> b s", s=seq_len)

def compute_grpo_no_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pi_ratio = torch.exp(policy_log_probs - old_log_probs) # (batch_size, seq_len)
    _, seq_len = policy_log_probs.shape
    advantages = _expand_advantages(advantages, seq_len) # (batch_size, seq_len)
    unclipped_loss = advantages * pi_ratio # (batch_size, seq_len)
    meta = {
        "cliped": torch.zeros_like(unclipped_loss, dtype=torch.bool),
        "ratio_extreme": pi_ratio >= 10.0,
    }
    return -unclipped_loss, meta

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pi_ratio = torch.exp(policy_log_probs - old_log_probs)
    _, seq_len = policy_log_probs.shape
    advantages = _expand_advantages(advantages, seq_len)
    v = pi_ratio * advantages
    v_clip = torch.clip(pi_ratio, min=1-cliprange, max=1+cliprange) * advantages

    meta = {"cliped": v > v_clip}
    return -torch.min(v, v_clip), meta


def compute_grpo_decoupled_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange_low: float,
    cliprange_high: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """DAPO-style decoupled clipping with asymmetric lower/upper bounds."""
    pi_ratio = torch.exp(policy_log_probs - old_log_probs)
    _, seq_len = policy_log_probs.shape
    advantages = _expand_advantages(advantages, seq_len)

    ratio_clipped = torch.clip(pi_ratio, min=1 - cliprange_low, max=1 + cliprange_high)
    v = pi_ratio * advantages
    v_clip = ratio_clipped * advantages
    low_hit = pi_ratio < (1 - cliprange_low)
    high_hit = pi_ratio > (1 + cliprange_high)
    return -torch.min(v, v_clip), {
        "cliped": low_hit | high_hit,
        "clip_low_hit": low_hit,
        "clip_high_hit": high_hit,
    }


def compute_gbpo_sign_aware_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    pos_cliprange_low: float,
    pos_cliprange_high: float,
    neg_cliprange_low: float,
    neg_cliprange_high: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """GBPO-lite sign-aware bounded clipping."""
    pi_ratio = torch.exp(policy_log_probs - old_log_probs)
    _, seq_len = policy_log_probs.shape
    advantages = _expand_advantages(advantages, seq_len)
    pos_mask = advantages >= 0

    lower = torch.where(
        pos_mask,
        torch.full_like(pi_ratio, 1 - pos_cliprange_low),
        torch.full_like(pi_ratio, 1 - neg_cliprange_low),
    )
    upper = torch.where(
        pos_mask,
        torch.full_like(pi_ratio, 1 + pos_cliprange_high),
        torch.full_like(pi_ratio, 1 + neg_cliprange_high),
    )

    ratio_clipped = torch.minimum(torch.maximum(pi_ratio, lower), upper)
    v = pi_ratio * advantages
    v_clip = ratio_clipped * advantages

    pos_low_hit = pos_mask & (pi_ratio < (1 - pos_cliprange_low))
    pos_high_hit = pos_mask & (pi_ratio > (1 + pos_cliprange_high))
    neg_low_hit = (~pos_mask) & (pi_ratio < (1 - neg_cliprange_low))
    neg_high_hit = (~pos_mask) & (pi_ratio > (1 + neg_cliprange_high))

    return -torch.min(v, v_clip), {
        "cliped": pos_low_hit | pos_high_hit | neg_low_hit | neg_high_hit,
        "gbpo_pos_clip_hit": pos_low_hit | pos_high_hit,
        "gbpo_neg_clip_hit": neg_low_hit | neg_high_hit,
        "gbpo_pos_clip_low_hit": pos_low_hit,
        "gbpo_pos_clip_high_hit": pos_high_hit,
        "gbpo_neg_clip_low_hit": neg_low_hit,
        "gbpo_neg_clip_high_hit": neg_high_hit,
    }

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: LossType,
    raw_rewards: torch.Tensor | None=None,
    advantages: torch.Tensor | None=None,
    old_log_probs: torch.Tensor | None=None,
    cliprange: float | None=None,
    cliprange_low: float | None=None,
    cliprange_high: float | None=None,
    gbpo_pos_cliprange_low: float | None=None,
    gbpo_pos_cliprange_high: float | None=None,
    gbpo_neg_cliprange_low: float | None=None,
    gbpo_neg_cliprange_high: float | None=None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "grpo_clip":
        assert advantages is not None
        assert old_log_probs is not None
        assert cliprange is not None
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    elif loss_type == "dapo_clip":
        assert advantages is not None
        assert old_log_probs is not None
        assert cliprange_low is not None
        assert cliprange_high is not None
        return compute_grpo_decoupled_clip_loss(
            advantages,
            policy_log_probs,
            old_log_probs,
            cliprange_low,
            cliprange_high,
        )
    elif loss_type == "gbpo_clip":
        assert advantages is not None
        assert old_log_probs is not None
        assert gbpo_pos_cliprange_low is not None
        assert gbpo_pos_cliprange_high is not None
        assert gbpo_neg_cliprange_low is not None
        assert gbpo_neg_cliprange_high is not None
        return compute_gbpo_sign_aware_clip_loss(
            advantages,
            policy_log_probs,
            old_log_probs,
            gbpo_pos_cliprange_low,
            gbpo_pos_cliprange_high,
            gbpo_neg_cliprange_low,
            gbpo_neg_cliprange_high,
        )
    elif loss_type == "grpo_no_clip":
        assert advantages is not None
        assert old_log_probs is not None
        assert cliprange is not None
        return compute_grpo_no_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    elif loss_type == "no_baseline":
        assert raw_rewards is not None
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def compute_token_approx_kl(
    policy_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    estimator: Literal["k1", "k2", "k3"] = "k3",
) -> torch.Tensor:
    """Estimate token-wise KL between policy and reference using action log-probs.

    Notes:
    - We only have log-probs on sampled actions, so this is an approximation.
    - k3 = exp(log_ratio) - 1 - log_ratio is non-negative and commonly used.
    """
    log_ratio = policy_log_probs - ref_log_probs
    if estimator == "k1":
        return log_ratio
    if estimator == "k2":
        return 0.5 * (log_ratio ** 2)
    return torch.exp(log_ratio) - 1.0 - log_ratio


def apply_kl_penalty_to_pg_loss(
    pg_loss: torch.Tensor,
    policy_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kl_beta: float,
    kl_estimator: Literal["k1", "k2", "k3"] = "k3",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Add KL penalty to a per-token policy-gradient loss tensor."""
    token_kl = compute_token_approx_kl(policy_log_probs, ref_log_probs, estimator=kl_estimator)
    kl_loss = kl_beta * token_kl
    total_loss = pg_loss + kl_loss
    return total_loss, {
        "token_kl": token_kl,
        "token_kl_loss": kl_loss,
        "token_pg_loss": pg_loss,
        "token_total_loss": total_loss,
    }


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None
) -> torch.Tensor:
    masked_tensor = torch.where(mask, tensor, torch.zeros_like(tensor))
    return torch.sum(masked_tensor, dim=dim) / torch.sum(mask, dim=dim)    # 进行了长度归一化, 为per token的loss


def grpo_microbatch_train_step(
        policy_log_probs: torch.Tensor, # (batch_size, seq_len)
        response_mask: torch.Tensor,    # (batch_size, seq_len)
        gradient_accumulation_steps: int,
        loss_type: LossType,
        raw_rewards: torch.Tensor | None=None,
        advantages: torch.Tensor | None=None,
        old_log_probs: torch.Tensor | None=None,
        cliprange: float | None=None,
        cliprange_low: float | None=None,
        cliprange_high: float | None=None,
        gbpo_pos_cliprange_low: float | None=None,
        gbpo_pos_cliprange_high: float | None=None,
        gbpo_neg_cliprange_low: float | None=None,
        gbpo_neg_cliprange_high: float | None=None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange,
        cliprange_low,
        cliprange_high,
        gbpo_pos_cliprange_low,
        gbpo_pos_cliprange_high,
        gbpo_neg_cliprange_low,
        gbpo_neg_cliprange_high,
    ) # (batch_size, seq_len)
    loss = masked_mean(
        loss,
        response_mask,
    ) # (batch_size, )
    loss /= gradient_accumulation_steps
    loss.backward()
    return loss, metadata


def grpo_microbatch_train_step_with_kl(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        loss_type: LossType,
        ref_log_probs: torch.Tensor,
        kl_beta: float,
        kl_estimator: Literal["k1", "k2", "k3"] = "k3",
        raw_rewards: torch.Tensor | None=None,
        advantages: torch.Tensor | None=None,
        old_log_probs: torch.Tensor | None=None,
        cliprange: float | None=None,
        cliprange_low: float | None=None,
        cliprange_high: float | None=None,
        gbpo_pos_cliprange_low: float | None=None,
        gbpo_pos_cliprange_high: float | None=None,
        gbpo_neg_cliprange_low: float | None=None,
        gbpo_neg_cliprange_high: float | None=None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Token-level aggregation variant with optional KL penalty."""
    pg_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange,
        cliprange_low,
        cliprange_high,
        gbpo_pos_cliprange_low,
        gbpo_pos_cliprange_high,
        gbpo_neg_cliprange_low,
        gbpo_neg_cliprange_high,
    )
    total_loss, kl_meta = apply_kl_penalty_to_pg_loss(
        pg_loss,
        policy_log_probs,
        ref_log_probs,
        kl_beta,
        kl_estimator
    )
    loss = masked_mean(total_loss, response_mask)
    loss /= gradient_accumulation_steps
    loss.backward()
    metadata.update(kl_meta)
    return loss, metadata


def grpo_microbatch_train_step_seq_level_loss(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        loss_type: LossType,
        raw_rewards: torch.Tensor | None=None,
        advantages: torch.Tensor | None=None,
        old_log_probs: torch.Tensor | None=None,
        cliprange: float | None=None,
        cliprange_low: float | None=None,
        cliprange_high: float | None=None,
        gbpo_pos_cliprange_low: float | None=None,
        gbpo_pos_cliprange_high: float | None=None,
        gbpo_neg_cliprange_low: float | None=None,
        gbpo_neg_cliprange_high: float | None=None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange,
        cliprange_low,
        cliprange_high,
        gbpo_pos_cliprange_low,
        gbpo_pos_cliprange_high,
        gbpo_neg_cliprange_low,
        gbpo_neg_cliprange_high,
    )
    constant = response_mask.sum(dim=-1).max().item()
    loss = masked_normalize(loss, response_mask, dim=-1, normalize_constant=constant).mean()
    loss /= gradient_accumulation_steps  # loss是相加的, 所以需要除以steps
    loss.backward()
    return loss, metadata


def grpo_microbatch_train_step_seq_level_loss_with_kl(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        loss_type: LossType,
        ref_log_probs: torch.Tensor,
        kl_beta: float,
        kl_estimator: Literal["k1", "k2", "k3"] = "k3",
        raw_rewards: torch.Tensor | None=None,
        advantages: torch.Tensor | None=None,
        old_log_probs: torch.Tensor | None=None,
        cliprange: float | None=None,
        cliprange_low: float | None=None,
        cliprange_high: float | None=None,
        gbpo_pos_cliprange_low: float | None=None,
        gbpo_pos_cliprange_high: float | None=None,
        gbpo_neg_cliprange_low: float | None=None,
        gbpo_neg_cliprange_high: float | None=None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Seq-level aggregation variant with optional KL penalty."""
    pg_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange,
        cliprange_low,
        cliprange_high,
        gbpo_pos_cliprange_low,
        gbpo_pos_cliprange_high,
        gbpo_neg_cliprange_low,
        gbpo_neg_cliprange_high,
    )
    total_loss, kl_meta = apply_kl_penalty_to_pg_loss(
        pg_loss,
        policy_log_probs,
        ref_log_probs,
        kl_beta,
        kl_estimator
    )
    # Keep the same sequence normalization behavior as existing seq-level training.
    constant = response_mask.sum(dim=-1).max().item()
    loss = masked_normalize(total_loss, response_mask, dim=-1, normalize_constant=constant).mean()
    loss /= gradient_accumulation_steps
    loss.backward()
    metadata.update(kl_meta)
    return loss, metadata
    
def gradient_clipping(model):
    params_gradients = []
    for param in model.parameters():
        if param.grad is not None:
            params_gradients.append(param.grad.data.flatten())
    grads = torch.cat(params_gradients)
    if torch.norm(grads) > 1.0:
        norm = torch.norm(grads)
        for param in model.parameters():
            param.grad.data /= norm
