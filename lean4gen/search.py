from __future__ import annotations
from dataclasses import dataclass, field
from .generator import LeanGenerator
from .repl import LeanRepl, TacticState


@dataclass
class SearchNode:
    state: TacticState
    tactics_so_far: list[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class SearchResult:
    success: bool
    proof: list[str]
    nodes_expanded: int


def format_tactic_prompt(statement: str, tactics_so_far: list[str], goal: str) -> str:
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
                return SearchResult(
                    success=True,
                    proof=node.tactics_so_far,
                    nodes_expanded=nodes_expanded,
                )
            goal = node.state.goals[0] if node.state.goals else ""
            prompt = format_tactic_prompt(statement, node.tactics_so_far, goal)
            proposed_tactics = generator.generate(prompt)[:candidates_per_step]
            for tactic in proposed_tactics:
                tactic = tactic.strip().split("\n")[0]
                if not tactic:
                    continue
                new_state = repl.run_tactic(node.state, tactic)
                nodes_expanded += 1
                if new_state.error:
                    continue
                candidates.append(
                    SearchNode(
                        state=new_state,
                        tactics_so_far=node.tactics_so_far + [tactic],
                        score=node.score
                        + (
                            1.0 if len(new_state.goals) < len(node.state.goals) else 0.0
                        ),
                    )
                )
        if not candidates:
            break
        candidates.sort(key=lambda n: (not n.state.done, len(n.state.goals), -n.score))
        beam = candidates[:beam_width]
        if beam[0].state.done:
            return SearchResult(
                success=True,
                proof=beam[0].tactics_so_far,
                nodes_expanded=nodes_expanded,
            )
    return SearchResult(success=False, proof=[], nodes_expanded=nodes_expanded)
