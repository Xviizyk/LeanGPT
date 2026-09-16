#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from leangpt.config import load_config
from leangpt.repl import LeanRepl
from leangpt.store import JsonlStore

SNIPPETS = [
    "theorem ok_proof : 1 + 1 = 2 := by decide",
    "theorem broken_proof : 1 + 1 = 3 := by decide",
    "theorem sorry_proof : 1 + 1 = 2 := by sorry",
]


def main() -> None:
    cfg = load_config()
    if not cfg.repl_bin or not cfg.lean_project_dir:
        raise SystemExit(
            "Укажи repl_bin / lean_project_dir в config.local.yaml или через REPL_BIN / LEAN_PROJECT_DIR"
        )
    store = JsonlStore("data/test_results.jsonl")
    with LeanRepl(repl_bin=cfg.repl_bin, project_dir=cfg.lean_project_dir) as repl:
        for code in SNIPPETS:
            result = repl.verify(code)
            store.add(
                statement=code,
                code=code,
                ok=result.ok,
                has_sorry=result.has_sorry,
                errors=result.errors,
            )
            status = "OK" if result.ok else "FAIL"
            print(
                f"[{status}] sorry={result.has_sorry} errors={result.errors}  <- {code}"
            )


if __name__ == "__main__":
    main()
