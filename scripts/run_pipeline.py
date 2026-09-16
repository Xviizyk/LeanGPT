#!/usr/bin/env python3
"""
Полный цикл самообучения с живым дашбордом (токены/сек, success rate,
текущий уровень curriculum). Личные пути (REPL, lake-проект, чекпоинты,
куда пушить) берутся из конфига — см. lean4gen/config.py — не хардкодятся
здесь, чтобы этот файл можно было держать в публичном репозитории.

Настройка конфига (в приватном lean4gen-client репозитории):
    export LEAN4GEN_CONFIG=/path/to/config.local.yaml
или через отдельные переменные окружения (REPL_BIN, LEAN_PROJECT_DIR, ...).

Запуск:
    python scripts/run_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lean4gen.config import load_config
from lean4gen.curriculum import curriculum_batch, should_advance
from lean4gen.dashboard import Dashboard
from lean4gen.generator import GenConfig, LeanGenerator
from lean4gen.repl import LeanRepl
from lean4gen.rl_train import RLConfig, grpo_step
from lean4gen.store import JsonlStore
from lean4gen.model import LeanGPT

N_ROUNDS_PER_LEVEL = 5
MAX_LEVEL = 4


def main() -> None:
    cfg = load_config()
    if not cfg.repl_bin or not cfg.lean_project_dir:
        raise SystemExit(
            "Не заданы repl_bin / lean_project_dir. Укажи их в config.local.yaml "
            "(см. lean4gen-client/config.local.yaml.example) или переменными "
            "окружения REPL_BIN / LEAN_PROJECT_DIR."
        )

    gen = LeanGenerator(GenConfig(num_candidates=8, checkpoint_path=cfg.checkpoint_path, tokenizer_dir=cfg.tokenizer_dir))
    gen.load()

    ref_model = LeanGPT(gen._model.cfg)
    ref_model.load_state_dict(gen._model.state_dict())
    ref_model.eval()

    store = JsonlStore(cfg.results_jsonl)
    level = 0

    import torch
    optim = torch.optim.AdamW(gen._model.parameters(), lr=RLConfig().lr)

    from tokenizers import ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer(f"{cfg.tokenizer_dir}/vocab.json", f"{cfg.tokenizer_dir}/merges.txt")

    with LeanRepl(repl_bin=cfg.repl_bin, project_dir=cfg.lean_project_dir, env_file=cfg.env_import) as repl, Dashboard() as dash:
        while level <= MAX_LEVEL:
            statements = curriculum_batch(level, n_per_level=20)

            for _ in range(N_ROUNDS_PER_LEVEL):
                for statement in statements:
                    grpo_step(statement, gen, ref_model, repl, optim, store, RLConfig(), tokenizer)

                    n_tok, dt = gen.last_gen_stats
                    dash.record_generation(n_tok, dt)

                    stats = store.stats()
                    ok = stats.get("success_rate", 0.0) > 0
                    dash.record_attempt(statement, level, ok)

            stats = store.stats()
            if should_advance(stats.get("success_rate", 0.0)):
                level += 1


if __name__ == "__main__":
    main()
