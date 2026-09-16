#!/usr/bin/env python3
import sys
from pathlib import Path
import torch
from tokenizers import ByteLevelBPETokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from leangpt.config import load_config
from leangpt.curriculum import curriculum_batch, eval_batch, should_advance
from leangpt.dashboard import Dashboard
from leangpt.eval import evaluate
from leangpt.generator import GenConfig, LeanGenerator
from leangpt.model import LeanGPT
from leangpt.repl_pool import ReplPool
from leangpt.rl_train import RLConfig, grpo_step
from leangpt.store import JsonlStore

N_ROUNDS_PER_LEVEL = 5
MAX_LEVEL = 4
EVAL_EVERY_N_ROUNDS = 2
REPL_POOL_SIZE = 4


def main() -> None:
    cfg = load_config()
    if not cfg.repl_bin or not cfg.lean_project_dir:
        raise SystemExit(
            "Не заданы repl_bin / lean_project_dir. Укажи их в config.local.yaml (см. LeanGPT-client/config.local.yaml.example) или переменными окружения REPL_BIN / LEAN_PROJECT_DIR."
        )
    gen = LeanGenerator(
        GenConfig(
            num_candidates=8,
            checkpoint_path=cfg.checkpoint_path,
            tokenizer_dir=cfg.tokenizer_dir,
        )
    )
    gen.load()
    ref_model = LeanGPT(gen._model.cfg)
    ref_model.load_state_dict(gen._model.state_dict())
    ref_model.eval()
    train_store = JsonlStore(cfg.results_jsonl)
    eval_store = JsonlStore(
        str(Path(cfg.results_jsonl).with_name("eval_results.jsonl"))
    )
    optim = torch.optim.AdamW(gen._model.parameters(), lr=RLConfig().lr)
    tokenizer = ByteLevelBPETokenizer(
        f"{cfg.tokenizer_dir}/vocab.json", f"{cfg.tokenizer_dir}/merges.txt"
    )
    level = 0
    with ReplPool(
        repl_bin=cfg.repl_bin,
        project_dir=cfg.lean_project_dir,
        pool_size=REPL_POOL_SIZE,
        env_file=cfg.env_import,
    ) as repl_pool, Dashboard() as dash:
        while level <= MAX_LEVEL:
            statements = curriculum_batch(level, n_per_level=20)
            advanced = False
            for round_i in range(N_ROUNDS_PER_LEVEL):
                for statement in statements:
                    grpo_step(
                        statement,
                        gen,
                        ref_model,
                        repl_pool,
                        optim,
                        train_store,
                        RLConfig(),
                        tokenizer,
                    )
                    n_tok, dt = gen.last_gen_stats
                    dash.record_generation(n_tok, dt)
                    dash.record_attempt(statement, level, ok=True)
                if round_i % EVAL_EVERY_N_ROUNDS == 0:
                    eval_statements = eval_batch(level, n=10)
                    eval_rate = evaluate(eval_statements, gen, repl_pool, eval_store)
                    print(
                        f"[eval] level={level} round={round_i} success_rate={eval_rate:.3f}"
                    )
                    if should_advance(eval_rate):
                        level += 1
                        advanced = True
                        break
            if not advanced:
                print(
                    f"[curriculum] level={level}: порог не достигнут за {N_ROUNDS_PER_LEVEL} раундов, повтор"
                )


if __name__ == "__main__":
    main()
