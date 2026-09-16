from __future__ import annotations
from typing import Iterable
from .generator import LeanGenerator, build_prompt
from .repl import LeanRepl
from .store import JsonlStore


def run(
    statements: Iterable[str],
    generator: LeanGenerator,
    repl: LeanRepl,
    store: JsonlStore,
    require_no_sorry: bool = True,
) -> None:
    total, saved = (0, 0)
    for statement in statements:
        prompt = build_prompt(statement)
        candidates = generator.generate(prompt)
        for candidate in candidates:
            total += 1
            full_code = (
                f"{statement} := {candidate}" if ":=" not in statement else candidate
            )
            result = repl.verify(full_code)
            keep = result.ok and (not require_no_sorry or not result.has_sorry)
            store.add(
                statement=statement,
                code=full_code,
                ok=result.ok,
                has_sorry=result.has_sorry,
                errors=result.errors,
            )
            if keep:
                saved += 1
                print(f"[OK]   {statement[:60]!r} -> сохранено")
            else:
                reason = "sorry" if result.has_sorry else "ошибка компиляции"
                print(f"[SKIP] {statement[:60]!r} -> {reason}")
    print(f"\nИтого: {saved}/{total} кандидатов прошли проверку и сохранены.")
