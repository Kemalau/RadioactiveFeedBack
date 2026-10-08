from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Appendix E: sample variance plus 1e-6, independently for each task."""
    if rewards.ndim != 2 or rewards.shape[1] < 2 or not torch.isfinite(rewards).all():
        raise ValueError("rewards must be a finite [tasks, responses>=2] tensor")
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    variance = centered.square().sum(dim=1, keepdim=True) / (rewards.shape[1] - 1)
    return centered / torch.sqrt(variance + 1e-6)


def grpo_loss(current: torch.Tensor, old: torch.Tensor, reference: torch.Tensor,
              mask: torch.Tensor, advantages: torch.Tensor, *, beta: float = 0.04,
              epsilon: float = 0.2) -> tuple[torch.Tensor, dict]:
    """Mean over responses of the mean completion-token GRPO loss."""
    if current.shape != old.shape or current.shape != reference.shape or current.shape != mask.shape:
        raise ValueError("log probabilities and token masks must have equal shapes")
    if current.ndim != 2 or advantages.shape != (current.shape[0],):
        raise ValueError("one advantage is required per response")
    if beta < 0 or epsilon < 0:
        raise ValueError("beta and clipping epsilon must be nonnegative")
    mask = mask.to(current.dtype)
    lengths = mask.sum(dim=1)
    if torch.any(lengths <= 0):
        raise ValueError("each response must have at least one completion token")
    active = mask.bool()
    current = torch.where(active, current, torch.zeros_like(current))
    old = torch.where(active, old, torch.zeros_like(old))
    reference = torch.where(active, reference, torch.zeros_like(reference))
    ratio = torch.exp(current - old.detach())
    advantage = advantages.detach().unsqueeze(1)
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - epsilon, 1 + epsilon) * advantage
    difference = reference.detach() - current
    kl = torch.exp(difference) - difference - 1
    token_loss = -torch.minimum(unclipped, clipped) + beta * kl
    loss = ((token_loss * mask).sum(dim=1) / lengths).mean()
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite GRPO loss")
    diagnostics = {
        "kl": ((kl * mask).sum(dim=1) / lengths).mean().detach().item(),
        "clip_fraction": (((ratio - 1).abs() > epsilon).to(mask.dtype) * mask).sum().detach().item()
                         / mask.sum().detach().item(),
    }
    return loss, diagnostics
