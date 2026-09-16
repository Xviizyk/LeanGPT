from __future__ import annotations
import math


def lr_schedule(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine)
