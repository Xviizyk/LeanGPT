"""
Оценка на held-out наборе (curriculum.eval_batch) — без шага обучения,
без записи в основной data/results.jsonl (иначе success_rate по всей
истории, который решает переход curriculum на следующий уровень,
подмешивал бы eval-примеры и искусственно завышался бы за счёт
переобучения на конкретных train-константах).
"""

from __future__ import annotations

from .generator import LeanGenerator, build_prompt
from .store import JsonlStore


def evaluate(statements: list[str], generator: LeanGenerator, repl, eval_store: JsonlStore) -> float:
    """Прогнать held-out теоремы через текущую модель, посчитать success rate.
    repl может быть LeanRepl или ReplPool (используется verify_many, если есть)."""
    prompts = [build_prompt(s) for s in statements]
    all_codes = []
    stmt_for_code = []

    for statement, prompt in zip(statements, prompts):
        candidates = generator.generate(prompt)[:1]  # на eval — один жадный/сэмплированный кандидат, без группы
        for cand in candidates:
            full_code = f"{statement} := {cand}" if ":=" not in statement else cand
            all_codes.append(full_code)
            stmt_for_code.append(statement)

    if hasattr(repl, "verify_many"):
        results = repl.verify_many(all_codes)
    else:
        results = [repl.verify(code) for code in all_codes]

    ok_count = 0
    for statement, code, result in zip(stmt_for_code, all_codes, results):
        eval_store.add(statement=statement, code=code, ok=result.ok, has_sorry=result.has_sorry, errors=result.errors)
        if result.ok and not result.has_sorry:
            ok_count += 1

    return ok_count / len(statements) if statements else 0.0
