#!/usr/bin/env python3
"""
Bot-дамп доказательств: пушит ВСЕ успешные попытки как есть, без дедупа,
без балансировки по тактикам, без ограничения на теорему — в отличие от
export_proofs.py (который курирует выборку). Здесь цель — просто копить
сырой поток в отдельном публичном репозитории; "мусор" в этом репо это ок,
это лог, а не курированная библиотека.

Инкрементальный: помнит, сколько строк results.jsonl уже обработано
(в .bot_state), при повторном запуске добавляет только новые записи.

Запуск (обычно из cron / после каждого раунда run_pipeline.py):
    python scripts/bot_dump_proofs.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lean4gen.config import load_config

STATE_FILE = ".bot_state"


def read_offset(state_path: Path) -> int:
    if state_path.exists():
        return int(state_path.read_text().strip() or "0")
    return 0


def write_offset(state_path: Path, offset: int) -> None:
    state_path.write_text(str(offset))


def dump_new(results_path: str, out_dir: str) -> int:
    results = Path(results_path)
    state_path = Path(out_dir) / STATE_FILE
    out_root = Path(out_dir) / "dump" / time.strftime("%Y-%m-%d")
    out_root.mkdir(parents=True, exist_ok=True)

    offset = read_offset(state_path)
    if not results.exists():
        return 0

    lines = results.read_text(encoding="utf-8").splitlines()
    new_lines = lines[offset:]

    count = 0
    ts = int(time.time())
    for i, line in enumerate(new_lines):
        rec = json.loads(line)
        if not rec.get("ok"):
            continue
        fname = out_root / f"{ts}_{offset + i}.lean"
        fname.write_text(rec["code"] + "\n", encoding="utf-8")
        count += 1

    write_offset(state_path, len(lines))
    return count


if __name__ == "__main__":
    cfg = load_config()
    out_dir = "proofs_dump_repo"  # рабочая копия репозитория доказательств (клонирован отдельно)
    n = dump_new(cfg.results_jsonl, out_dir)
    print(f"Добавлено новых файлов: {n}")
