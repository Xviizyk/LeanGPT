from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from tokenizers import ByteLevelBPETokenizer

from .generator import LeanGenerator, build_prompt
from .model import LeanGPT
from .store import JsonlStore

EOS_TOKEN = "<eos>"

_CHEAT = re.compile(r"\b(sorry|sorryAx|admit|axiom|native_decide)\b")


@dataclass
class RLConfig:
    group_size: int = 8
    lr: float = 1e-5
    kl_coef: float = 0.02
    clip_grad: float = 1.0
    length_penalty: float = 0.01
    min_proof_reward: float = 0.1

@dataclass
class StepResult:
    mean_reward: float
    n_proved: int
    n_total: int

    @property
    def ok(self) -> bool:
        return self.n_proved > 0

    def __float__(self) -> float:
        return self.mean_reward


def is_cheat(code: str) -> bool:
    return bool(_CHEAT.search(code))


def is_real_proof(ok: bool, has_sorry: bool, code: str) -> bool:
    """Единое правило «доказано». Используй и в eval.py."""
    return bool(ok) and not has_sorry and bool(code.strip()) and not is_cheat(code)


def reward_fn(
    ok: bool,
    has_sorry: bool,
    n_tactics: int = 0,
    length_penalty: float = 0.0,
    min_reward: float = 0.1,
) -> float:
    if not ok or has_sorry:
        return 0.0
    return max(min_reward, 1.0 - length_penalty * n_tactics)


def count_tactics(code: str) -> int:
    return max(1, code.count("\n") + code.count(";") + 1)


def grpo_step(
    statement: str,
    generator: LeanGenerator,
    ref_model: LeanGPT,
    repl,
    optim: torch.optim.Optimizer,
    store: JsonlStore,
    cfg: RLConfig,
    tokenizer: ByteLevelBPETokenizer,
) -> StepResult:
    prompt = build_prompt(statement)
    candidates = generator.generate(prompt)[: cfg.group_size]
    full_codes = [
        f"{statement} := {cand}" if ":=" not in statement else cand
        for cand in candidates
    ]

    if hasattr(repl, "verify_many"):
        verify_results = repl.verify_many(full_codes)
    else:
        verify_results = [repl.verify(code) for code in full_codes]

    rewards: list[float] = []
    n_proved = 0
    for cand, full_code, result in zip(candidates, full_codes, verify_results):
        has_sorry = bool(result.has_sorry) or is_cheat(cand)
        ok = bool(result.ok) and bool(cand.strip())
        r = reward_fn(
            ok,
            has_sorry,
            count_tactics(cand),
            cfg.length_penalty,
            cfg.min_proof_reward,
        )
        rewards.append(r)
        n_proved += int(ok and not has_sorry)
        store.add(
            statement=statement,
            code=full_code,
            ok=ok,
            has_sorry=has_sorry,
            errors=result.errors,
        )

    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    mean_r = rewards_t.mean().item()
    step_res = StepResult(mean_r, n_proved, len(candidates))

    if rewards_t.std() < 1e-6:
        return step_res

    advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-6)

    device = next(generator._model.parameters()).device
    prompt_ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)

    optim.zero_grad()
    n = len(candidates)
    trained = 0
    for cand, adv in zip(candidates, advantages):
        ids = tokenizer.encode(cand).ids
        if eos_id is not None:
            ids = ids + [eos_id]
        if not ids:
            continue
        cand_ids = torch.tensor([ids], device=device)

        logp = generator._model.logprob_of_continuation(prompt_ids, cand_ids)
        with torch.no_grad():
            ref_logp = ref_model.logprob_of_continuation(prompt_ids, cand_ids)
        kl = logp - ref_logp

        loss = (-adv.to(device) * logp + cfg.kl_coef * kl).sum() / n
        loss.backward()
        trained += 1

    if trained:
        torch.nn.utils.clip_grad_norm_(generator._model.parameters(), cfg.clip_grad)
        optim.step()

    return step_res


def run_grpo(
    statements: list[str],
    generator: LeanGenerator,
    repl,
    store: JsonlStore,
    tokenizer_dir: str = "data/tokenizer",
    cfg: RLConfig = RLConfig(),
    n_rounds: int = 1,
) -> None:
    tokenizer = ByteLevelBPETokenizer(
        f"{tokenizer_dir}/vocab.json", f"{tokenizer_dir}/merges.txt"
    )

    ref_model = LeanGPT(generator._model.cfg).to(
        next(generator._model.parameters()).device
    )
    ref_model.load_state_dict(generator._model.state_dict())
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    optim = torch.optim.AdamW(generator._model.parameters(), lr=cfg.lr)

    for round_i in range(n_rounds):
        results = [
            grpo_step(s, generator, ref_model, repl, optim, store, cfg, tokenizer)
            for s in statements
        ]
        avg = sum(r.mean_reward for r in results) / len(results) if results else 0.0
        proved = sum(r.n_proved for r in results)
        total = sum(r.n_total for r in results)
        print(
            f"[GRPO] Round {round_i}: average reward = {avg:.3f}, proved {proved}/{total}"
        )