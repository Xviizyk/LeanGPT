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


def level_0(n: int = 20, seed_offset: int = 0) -> list[str]:
    out = []
    for i in range(n):
        a, b = _det_pair(i, seed_offset)
        out.append(f"theorem const_add_{a}_{b} : {a} + {b} = {a + b}")
    return out


def _det_pair(i: int, range_base: int = 0) -> tuple[int, int]:
    """Детерминированная пара по индексу, сдвинутая на range_base — нужно,
    чтобы eval-набор был воспроизводим и гарантированно не пересекался с
    train-набором по значениям (не только по индексу i, ведь имя теоремы
    строится из a/b, а не из i). Сдвиг диапазона значений (а не просто i)
    даёт непересекающиеся множества пар при разных range_base."""
    return (range_base + i % 21, range_base + (i * 7) % 21)


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
        k = (idx % 5) + 1  # детерминированно вместо random.randint — нужно для eval-набора
        elems = ", ".join(str(x) for x in range(k))
        out.append(f"theorem list_len_{idx} : [{elems}].length = {k}")
    return out


def level_4(n: int = 20) -> list[str]:
    templates = [
        "theorem nat_add_zero_ind_{i} (n : Nat) : n + 0 = n",
        "theorem nat_succ_add_{i} (n m : Nat) : n.succ + m = (n + m).succ",
    ]
    return [random.choice(templates).format(i=i) for i in range(n)]


LEVELS = [level_0, level_1, level_2, level_3, level_4]

# Уровни с настоящими вариациями по значениям (не только по имени теоремы) —
# для них есть смысл держать непересекающийся eval-набор через seed_offset.
# level_1/level_2/level_4 — универсально-квантифицированные тождества
# (add_comm для ЛЮБЫХ a b), там нет "конкретного инстанса", который можно
# было бы держать в секрете — переобучиться там особо не на чем.
_EVAL_OFFSET = 10_000  # eval использует индексы, гарантированно не пересекающиеся с train


def curriculum_batch(current_level: int, n_per_level: int = 20) -> list[str]:
    """Вернуть смесь: в основном текущий уровень + немного предыдущих
    (чтобы не забывать простые случаи) — классический curriculum-replay."""
    batch: list[str] = []
    for lvl in range(current_level + 1):
        count = n_per_level if lvl == current_level else max(1, n_per_level // 4)
        batch.extend(LEVELS[lvl](count))
    random.shuffle(batch)
    return batch


def eval_batch(current_level: int, n: int = 10) -> list[str]:
    """Фиксированный held-out набор для оценки — НЕ используется для
    обучения. Для level_0/level_3 (где есть настоящие вариации по
    значениям) берём непересекающийся диапазон индексов через
    seed_offset. Для остальных уровней теоремы универсально-
    квантифицированы (add_comm для любых a,b) — там "инстанс" один и тот
    же независимо от имени, отдельный eval не нужен, переиспользуем как есть."""
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
    """Критерий перехода на следующий уровень сложности."""
    return success_rate >= threshold
