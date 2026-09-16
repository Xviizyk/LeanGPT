from __future__ import annotations
from dataclasses import dataclass
import torch
from tokenizers import ByteLevelBPETokenizer
from .generator import GenConfig, LeanGenerator, build_prompt
from .model import LeanGPT, ModelConfig
from .store import JsonlStore


@dataclass
class RLConfig:
    group_size: int = 8
    lr: float = 1e-05
    kl_coef: float = 0.02
    clip_grad: float = 1.0
    length_penalty: float = 0.01


def reward_fn(
    ok: bool, has_sorry: bool, n_tactics: int = 0, length_penalty: float = 0.0
) -> float:
    if ok and (not has_sorry):
        base = 1.0
    elif ok and has_sorry:
        base = 0.2
    else:
        return 0.0
    return max(0.0, base - length_penalty * n_tactics)


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
) -> float:
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
    rewards = []
    for cand, full_code, result in zip(candidates, full_codes, verify_results):
        n_tactics = count_tactics(full_code)
        r = reward_fn(result.ok, result.has_sorry, n_tactics, cfg.length_penalty)
        rewards.append(r)
        store.add(
            statement=statement,
            code=full_code,
            ok=result.ok,
            has_sorry=result.has_sorry,
            errors=result.errors,
        )
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    if rewards_t.std() < 1e-06:
        return rewards_t.mean().item()
    advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-06)
    device = next(generator._model.parameters()).device
    prompt_ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
    total_loss = torch.tensor(0.0, device=device)
    for cand, adv in zip(candidates, advantages):
        cand_ids = torch.tensor([tokenizer.encode(cand).ids], device=device)
        if cand_ids.shape[1] == 0:
            continue
        logp = generator._model.logprob_of_continuation(prompt_ids, cand_ids)
        with torch.no_grad():
            ref_logp = ref_model.logprob_of_continuation(prompt_ids, cand_ids)
        kl = logp - ref_logp
        total_loss = total_loss - adv.to(device) * logp + cfg.kl_coef * kl
    total_loss = total_loss / len(candidates)
    optim.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(generator._model.parameters(), cfg.clip_grad)
    optim.step()
    return rewards_t.mean().item()


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
        round_rewards = []
        for statement in statements:
            r = grpo_step(
                statement, generator, ref_model, repl, optim, store, cfg, tokenizer
            )
            round_rewards.append(r)
        avg = sum(round_rewards) / len(round_rewards) if round_rewards else 0.0
        print(f"[GRPO] раунд {round_i}: средний reward = {avg:.3f}")
