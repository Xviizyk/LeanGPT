"""
Best-first поиск доказательства: на каждом шаге модель предлагает
несколько кандидатов-тактик для текущей цели, REPL проверяет каждую
и возвращает новое состояние (goals) — сигнал "продвинулись/нет"
получаем на каждом шаге, а не только в самом конце.

Это заменяет "generate the whole proof and pray" на пошаговый поиск,
похожий по духу на ReProver/AlphaProof (без готовой LLM — с той же
LeanGPT, но применяемой рекуррентно к all-goals-so-far).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .generator import LeanGenerator
from .repl import LeanRepl, TacticState


@dataclass
class SearchNode:
    state: TacticState
    tactics_so_far: list[str] = field(default_factory=list)
    score: float = 0.0  # накопленный log-prob или число успешных шагов


@dataclass
class SearchResult:
    success: bool
    proof: list[str]
    nodes_expanded: int


def format_tactic_prompt(statement: str, tactics_so_far: list[str], goal: str) -> str:
    """Промпт для генерации следующей тактики: контекст (что уже сделано) + текущая цель."""
    history = "\n".join(tactics_so_far)
    return f"{statement} := by\n{history}\n-- goal: {goal}\n"


def beam_search_proof(
    statement: str,
    generator: LeanGenerator,
    repl: LeanRepl,
    beam_width: int = 4,
    max_steps: int = 20,
    candidates_per_step: int = 8,
) -> SearchResult:
    """
    Best-first поиск с шириной луча beam_width.
    На каждом шаге: для каждого узла в луче генерируем candidates_per_step
    тактик, проверяем их в REPL, оставляем beam_width лучших по числу
    оставшихся целей (меньше — лучше), приоритет успешным (done=True).
    """
    root_state = repl.open_goal(statement)
    if root_state.error:
        return SearchResult(success=False, proof=[], nodes_expanded=0)
    if root_state.done:
        return SearchResult(success=True, proof=[], nodes_expanded=0)

    beam = [SearchNode(state=root_state)]
    nodes_expanded = 0

    for step in range(max_steps):
        candidates: list[SearchNode] = []

        for node in beam:
            if node.state.done:
                return SearchResult(success=True, proof=node.tactics_so_far, nodes_expanded=nodes_expanded)

            goal = node.state.goals[0] if node.state.goals else ""
            prompt = format_tactic_prompt(statement, node.tactics_so_far, goal)
            proposed_tactics = generator.generate(prompt)[:candidates_per_step]

            for tactic in proposed_tactics:
                tactic = tactic.strip().split("\n")[0]  # берём первую строку как одну тактику
                if not tactic:
                    continue
                new_state = repl.run_tactic(node.state, tactic)
                nodes_expanded += 1

                if new_state.error:
                    continue  # тактика не применилась — отбрасываем

                candidates.append(
                    SearchNode(
                        state=new_state,
                        tactics_so_far=node.tactics_so_far + [tactic],
                        score=node.score + (1.0 if len(new_state.goals) < len(node.state.goals) else 0.0),
                    )
                )

        if not candidates:
            break  # луч вымер — ни одна тактика ни на одном узле не применилась

        # успешные узлы (done=True) — сразу в приоритет
        candidates.sort(key=lambda n: (not n.state.done, len(n.state.goals), -n.score))
        beam = candidates[:beam_width]

        if beam[0].state.done:
            return SearchResult(success=True, proof=beam[0].tactics_so_far, nodes_expanded=nodes_expanded)

    return SearchResult(success=False, proof=[], nodes_expanded=nodes_expanded)
