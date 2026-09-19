#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import torch
from tokenizers import ByteLevelBPETokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from leangpt.config import load_config
from leangpt.curriculum import curriculum_batch, eval_batch, should_advance
from leangpt.dashboard import Dashboard
from leangpt.eval import evaluate
from leangpt.gen_dump import GenDump
from leangpt.generator import GenConfig, LeanGenerator
from leangpt.model import LeanGPT
from leangpt.repl_pool import ReplPool
from leangpt.rl_train import RLConfig, grpo_step
from leangpt.store import JsonlStore

N_ROUNDS_PER_LEVEL = 5
MAX_LEVEL = 4
EVAL_EVERY_N_ROUNDS = 2
REPL_POOL_SIZE = 4
SAVE_EVERY_STEPS = 50


def save_ckpt(
    path: Path,
    model: LeanGPT,
    optim: torch.optim.Optimizer,
    level: int,
    round_i: int,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "config": model.cfg,  # нужно для LeanGenerator.load()
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "level": level,
            "round": round_i,
            "step": step,
        },
        tmp,
    )
    os.replace(tmp, path)
    print(f"[ckpt] сохранён {path} (level={level} round={round_i} step={step})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="не подхватывать rl_latest.pt, начать RL с исходного претрейна",
    )
    args = parser.parse_args()

    cfg = load_config()
    if not cfg.repl_bin or not cfg.lean_project_dir:
        raise SystemExit(
            "Не заданы repl_bin / lean_project_dir. Укажи их в config.local.yaml (см. LeanGPT-client/config.local.yaml.example) или переменными окружения REPL_BIN / LEAN_PROJECT_DIR."
        )

    ckpt_dir = Path(cfg.checkpoint_path).parent
    rl_latest = ckpt_dir / "rl_latest.pt"

    gen = LeanGenerator(
        GenConfig(
            num_candidates=8,
            checkpoint_path=cfg.checkpoint_path,
            tokenizer_dir=cfg.tokenizer_dir,
        )
    )
    gen.load()
    dump = GenDump(gen, Path(cfg.results_jsonl).parent / "generated")

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
    step = 0
    round_i = 0
    if rl_latest.exists() and not args.fresh:
        state = torch.load(rl_latest, map_location=gen.cfg.device, weights_only=False)
        gen._model.load_state_dict(state["model"])
        optim.load_state_dict(state["optim"])
        level = int(state.get("level", 0))
        step = int(state.get("step", 0))
        print(f"[ckpt] продолжаю с {rl_latest} (level={level} step={step})")
    elif rl_latest.exists():
        print(f"[ckpt] --fresh: {rl_latest} игнорирую (при первом сохранении будет перезаписан)")

    with ReplPool(
        repl_bin=cfg.repl_bin,
        project_dir=cfg.lean_project_dir,
        pool_size=REPL_POOL_SIZE,
        env_file=cfg.env_import,
    ) as repl_pool, Dashboard() as dash:
        try:
            while level <= MAX_LEVEL:
                dump.level = level
                statements = curriculum_batch(level, n_per_level=20)
                advanced = False
                for round_i in range(N_ROUNDS_PER_LEVEL):
                    proved = total = 0
                    for statement in statements:
                        res = grpo_step(
                            statement,
                            gen,
                            ref_model,
                            repl_pool,
                            optim,
                            train_store,
                            RLConfig(),
                            tokenizer,
                        )
                        step += 1
                        proved += res.n_proved
                        total += res.n_total
                        n_tok, dt = gen.last_gen_stats
                        dash.record_generation(n_tok, dt)
                        dash.record_attempt(statement, level, ok=res.ok)
                        if step % SAVE_EVERY_STEPS == 0:
                            save_ckpt(rl_latest, gen._model, optim, level, round_i, step)
                    print(
                        f"[train] level={level} round={round_i} proved={proved}/{total}"
                    )
                    if round_i % EVAL_EVERY_N_ROUNDS == 0:
                        eval_statements = eval_batch(level, n=10)
                        eval_rate = evaluate(
                            eval_statements, gen, repl_pool, eval_store
                        )
                        print(
                            f"[eval] level={level} round={round_i} success_rate={eval_rate:.3f}"
                        )
                        save_ckpt(rl_latest, gen._model, optim, level, round_i, step)
                        if should_advance(eval_rate):
                            save_ckpt(
                                ckpt_dir / f"rl_level{level}.pt",
                                gen._model,
                                optim,
                                level,
                                round_i,
                                step,
                            )
                            level += 1
                            advanced = True
                            break
                if not advanced:
                    print(
                        f"[curriculum] level={level}: порог не достигнут за {N_ROUNDS_PER_LEVEL} раундов, повтор"
                    )
        finally:
            save_ckpt(rl_latest, gen._model, optim, level, round_i, step)


if __name__ == "__main__":
    main()