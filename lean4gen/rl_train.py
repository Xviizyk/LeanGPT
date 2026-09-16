"""
RL-дообучение напрямую по сигналу от REPL — без промежуточного шага
"сохранить успешные примеры как текст -> переобучить как LM".

Схема — GRPO-lite (упрощённо, без критика/value-модели, как в DeepSeekMath/
DeepSeek-Prover): для каждой формулировки генерируем группу из G кандидатов,
считаем награду каждого (успех=1 либо частичный сигнал от beam_search),
нормализуем награду внутри группы (advantage = r - mean(r)) и делаем шаг
градиента, увеличивающий log-prob кандидатов с advantage > 0 и уменьшающий
для advantage < 0.

Это быстрее итерируется, чем цикл "generate -> save -> retrain from scratch",
и даёт куда более точный сигнал, чем бинарный success/fail на уровне текста.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenizers import ByteLevelBPETokenizer

from .generator import GenConfig, LeanGenerator, build_prompt
from .model import LeanGPT, ModelConfig
from .repl import LeanRepl
from .store import JsonlStore


@dataclass
class RLConfig:
    group_size: int = 8       # G кандидатов на одну формулировку (как в GRPO)
    lr: float = 1e-5
    kl_coef: float = 0.02     # штраф за отклонение от референс-модели (анти-collapse)
    clip_grad: float = 1.0


def reward_fn(ok: bool, has_sorry: bool) -> float:
    if ok and not has_sorry:
        return 1.0
    if ok and has_sorry:
        return 0.2  # формально скомпилировалось, но это не настоящее доказательство
    return 0.0


def grpo_step(
    statement: str,
    generator: LeanGenerator,
    ref_model: LeanGPT,
    repl: LeanRepl,
    optim: torch.optim.Optimizer,
    store: JsonlStore,
    cfg: RLConfig,
    tokenizer: ByteLevelBPETokenizer,
) -> float:
    """Один шаг GRPO на одной формулировке. Возвращает средний reward группы
    (удобно логировать как метрику прогресса curriculum)."""
    prompt = build_prompt(statement)
    candidates = generator.generate(prompt)[: cfg.group_size]

    rewards = []
    for cand in candidates:
        full_code = f"{statement} := {cand}" if ":=" not in statement else cand
        result = repl.verify(full_code)
        r = reward_fn(result.ok, result.has_sorry)
        rewards.append(r)
        store.add(statement=statement, code=full_code, ok=result.ok, has_sorry=result.has_sorry, errors=result.errors)

    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    if rewards_t.std() < 1e-6:
        return rewards_t.mean().item()  # вся группа одинакова — нечему учиться на этом шаге

    advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-6)

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
        kl = logp - ref_logp  # приближение KL(policy || ref) на сэмпле

        # policy gradient: -advantage * logp, плюс KL-штраф против коллапса в узкое распределение
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
    repl: LeanRepl,
    store: JsonlStore,
    tokenizer_dir: str = "data/tokenizer",
    cfg: RLConfig = RLConfig(),
    n_rounds: int = 1,
) -> None:
    tokenizer = ByteLevelBPETokenizer(f"{tokenizer_dir}/vocab.json", f"{tokenizer_dir}/merges.txt")

    # референс-модель — замороженная копия для KL-штрафа (анти-collapse)
    ref_model = LeanGPT(generator._model.cfg).to(next(generator._model.parameters()).device)
    ref_model.load_state_dict(generator._model.state_dict())
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    optim = torch.optim.AdamW(generator._model.parameters(), lr=cfg.lr)

    for round_i in range(n_rounds):
        round_rewards = []
        for statement in statements:
            r = grpo_step(statement, generator, ref_model, repl, optim, store, cfg, tokenizer)
            round_rewards.append(r)
        avg = sum(round_rewards) / len(round_rewards) if round_rewards else 0.0
        print(f"[GRPO] раунд {round_i}: средний reward = {avg:.3f}")
