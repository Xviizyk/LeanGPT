#!/usr/bin/env python3
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lean4gen.store import JsonlStore, tactic_signature


def guess_level(statement: str) -> str:
    name_match = re.search("theorem\\s+(\\S+)", statement)
    name = name_match.group(1) if name_match else ""
    if name.startswith("const_"):
        return "Level0_Constants"
    if name.startswith(("add_comm", "add_zero", "zero_add", "mul_one")):
        return "Level1_Arithmetic"
    if name.startswith(("and_comm", "or_comm", "not_not")):
        return "Level2_BoolLogic"
    if name.startswith("list_len"):
        return "Level3_Lists"
    if name.startswith(("nat_add_zero_ind", "nat_succ_add")):
        return "Level4_Induction"
    return "Custom"


def safe_filename(statement: str) -> str:
    name_match = re.search("theorem\\s+(\\S+)", statement)
    name = name_match.group(1) if name_match else "unnamed"
    return re.sub("[^A-Za-z0-9_]", "_", name)


def export(
    results_path: str, out_dir: str, max_per_signature: int, max_per_statement: int
) -> None:
    store = JsonlStore(results_path)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    name_counts: dict[str, int] = defaultdict(int)
    level_counts: dict[str, int] = defaultdict(int)
    total = 0
    for rec in store.iter_balanced(
        max_per_signature=max_per_signature, max_per_statement=max_per_statement
    ):
        level = guess_level(rec["statement"])
        level_dir = out_root / level
        level_dir.mkdir(parents=True, exist_ok=True)
        base_name = safe_filename(rec["statement"])
        name_counts[base_name] += 1
        suffix = "" if name_counts[base_name] == 1 else f"__v{name_counts[base_name]}"
        file_path = level_dir / f"{base_name}{suffix}.lean"
        file_path.write_text(rec["code"] + "\n", encoding="utf-8")
        level_counts[level] += 1
        total += 1
    stats = store.stats()
    readme_lines = [
        "# LeanGPT proof library",
        "",
        "Автоматически сгенерировано и доказано LeanGPT (см. основной репозиторий lean4gen).",
        "",
        f"- Всего экспортировано доказательств: {total}",
        f"- Общий success_rate по истории генераций: {stats.get('success_rate', 0):.3f}",
        "",
        "## По уровням сложности",
        "",
    ]
    for level, count in sorted(level_counts.items()):
        readme_lines.append(f"- {level}: {count}")
    (out_root / "README.md").write_text(
        "\n".join(readme_lines) + "\n", encoding="utf-8"
    )
    print(f"Экспортировано {total} файлов в {out_root}")
    for level, count in sorted(level_counts.items()):
        print(f"  {level}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="data/results.jsonl")
    parser.add_argument("--out", default="proof_library")
    parser.add_argument("--max-per-signature", type=int, default=200)
    parser.add_argument("--max-per-statement", type=int, default=20)
    args = parser.parse_args()
    export(args.results, args.out, args.max_per_signature, args.max_per_statement)
