import torch
import torch.nn.functional as F


def compute_masked_token_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_response_mask: torch.Tensor,
    teacher_response_mask: torch.Tensor,
    kl_coef: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute token-level KL(student || teacher) on aligned response tokens only."""
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)

    # Teacher/student prompts can differ (e.g., privileged hint in OPSD), so sequence
    # lengths are not guaranteed to match. We align only response-token positions.
    batch_size = student_log_probs.shape[0]
    total_kl = torch.zeros((), device=student_log_probs.device, dtype=student_log_probs.dtype)
    total_tokens = torch.zeros((), device=student_log_probs.device, dtype=student_log_probs.dtype)
    clipped_count = 0

    for b in range(batch_size):
        s_lp = student_log_probs[b][student_response_mask[b]]
        t_p = teacher_probs[b][teacher_response_mask[b]]
        n = min(s_lp.shape[0], t_p.shape[0])
        if n <= 0:
            continue
        if s_lp.shape[0] != t_p.shape[0]:
            clipped_count += 1
        token_kl_b = F.kl_div(s_lp[:n], t_p[:n], reduction="none").sum(dim=-1)
        total_kl = total_kl + token_kl_b.sum()
        total_tokens = total_tokens + torch.tensor(float(n), device=total_tokens.device, dtype=total_tokens.dtype)

    kl_mean = total_kl / total_tokens.clamp(min=1.0)
    loss = kl_coef * kl_mean
    meta = {
        "kl_mean": kl_mean.detach(),
        "aligned_tokens": total_tokens.detach(),
        "length_mismatch_count": torch.tensor(float(clipped_count), device=kl_mean.device),
    }
    return loss, meta
