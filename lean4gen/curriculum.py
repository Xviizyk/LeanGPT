"""
Curriculum: генератор синтетических формулировок теорем нарастающей
сложности. Нужен, потому что модель "с нуля" почти никогда не решит
реальные mathlib-теоремы на старте — датасет успешных примеров будет
пустым. Начинаем с тривиального и постепенно усложняем.

Уровни:
  0 — rfl/decide на константах: "1 + 1 = 2"
  1 — простая арифметика с переменными: "a + b = b + a" (Nat)
  2 — булева логика: "p && q = q && p"
  3 — списки: "[a, b].length = 2"
  4 — индукция: "n + 0 = n" через induction/simp
"""

from __future__ import annotations

import random


def level_0(n: int = 20) -> list[str]:
    out = []
    for _ in range(n):
        a, b = random.randint(0, 20), random.randint(0, 20)
        out.append(f"theorem const_add_{a}_{b} : {a} + {b} = {a + b}")
    return out


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


def level_3(n: int = 20) -> list[str]:
    out = []
    for i in range(n):
        k = random.randint(1, 5)
        elems = ", ".join(str(x) for x in range(k))
        out.append(f"theorem list_len_{i} : [{elems}].length = {k}")
    return out


def level_4(n: int = 20) -> list[str]:
    templates = [
        "theorem nat_add_zero_ind_{i} (n : Nat) : n + 0 = n",
        "theorem nat_succ_add_{i} (n m : Nat) : n.succ + m = (n + m).succ",
    ]
    return [random.choice(templates).format(i=i) for i in range(n)]


LEVELS = [level_0, level_1, level_2, level_3, level_4]


def curriculum_batch(current_level: int, n_per_level: int = 20) -> list[str]:
    """Вернуть смесь: в основном текущий уровень + немного предыдущих
    (чтобы не забывать простые случаи) — классический curriculum-replay."""
    batch: list[str] = []
    for lvl in range(current_level + 1):
        count = n_per_level if lvl == current_level else max(1, n_per_level // 4)
        batch.extend(LEVELS[lvl](count))
    random.shuffle(batch)
    return batch


def should_advance(success_rate: float, threshold: float = 0.6) -> bool:
    """Критерий перехода на следующий уровень сложности."""
    return success_rate >= threshold
