from __future__ import annotations
import random


def level_0(n: int = 20, seed_offset: int = 0) -> list[str]:
    out = []
    for i in range(n):
        a, b = _det_pair(i, seed_offset)
        out.append(f"theorem const_add_{a}_{b} : {a} + {b} = {a + b}")
    return out


def _det_pair(i: int, range_base: int = 0) -> tuple[int, int]:
    return (range_base + i % 21, range_base + i * 7 % 21)


def level_1(n: int = 20) -> list[str]:
    templates = [
        "theorem add_comm_{i} (a b : Nat) : a + b = b + a",
        "theorem add_zero_{i} (a : Nat) : a + 0 = a",
        "theorem zero_add_{i} (a : Nat) : 0 + a = a",
        "theorem mul_one_{i} (a : Nat) : a * 1 = a",
    ]
    return [random.choice(templates).format(i=i) for i in range(n)]


def level_2(n: int = 20) -> list[str]:
    templates = [
        "theorem and_comm_{i} (p q : Bool) : (p && q) = (q && p)",
        "theorem or_comm_{i} (p q : Bool) : (p || q) = (q || p)",
        "theorem not_not_{i} (p : Bool) : !!p = p",
    ]
    return [random.choice(templates).format(i=i) for i in range(n)]


def level_3(n: int = 20, seed_offset: int = 0) -> list[str]:
    out = []
    for i in range(n):
        idx = i + seed_offset
        k = idx % 5 + 1
        elems = ", ".join((str(x) for x in range(k)))
        out.append(f"theorem list_len_{idx} : [{elems}].length = {k}")
    return out


def level_4(n: int = 20) -> list[str]:
    templates = [
        "theorem nat_add_zero_ind_{i} (n : Nat) : n + 0 = n",
        "theorem nat_succ_add_{i} (n m : Nat) : n.succ + m = (n + m).succ",
    ]
    return [random.choice(templates).format(i=i) for i in range(n)]


LEVELS = [level_0, level_1, level_2, level_3, level_4]
_EVAL_OFFSET = 10000


def curriculum_batch(current_level: int, n_per_level: int = 20) -> list[str]:
    batch: list[str] = []
    for lvl in range(current_level + 1):
        count = n_per_level if lvl == current_level else max(1, n_per_level // 4)
        batch.extend(LEVELS[lvl](count))
    random.shuffle(batch)
    return batch


def eval_batch(current_level: int, n: int = 10) -> list[str]:
    batch: list[str] = []
    for lvl in range(current_level + 1):
        if lvl == 0:
            batch.extend(level_0(n, seed_offset=_EVAL_OFFSET))
        elif lvl == 3:
            batch.extend(level_3(n, seed_offset=_EVAL_OFFSET))
        else:
            batch.extend(LEVELS[lvl](n))
    return batch


def should_advance(success_rate: float, threshold: float = 0.6) -> bool:
    return success_rate >= threshold
