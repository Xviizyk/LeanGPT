#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from leangpt.config import load_config

STATE_FILE = ".bot_state"


def read_offset(state_path: Path) -> int:
    if state_path.exists():
        return int(state_path.read_text().strip() or "0")
    return 0


def write_offset(state_path: Path, offset: int) -> None:
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp_path.write_text(str(offset))
    tmp_path.replace(state_path)


def dump_new(results_path: str, out_dir: str) -> int:
    results = Path(results_path)
    state_path = Path(out_dir) / STATE_FILE
    out_root = Path(out_dir) / "dump" / time.strftime("%Y-%m-%d")
    out_root.mkdir(parents=True, exist_ok=True)
    offset = read_offset(state_path)
    if not results.exists():
        return 0
    raw = results.read_text(encoding="utf-8")
    ends_complete = raw.endswith("\n")
    lines = raw.splitlines()
    if not ends_complete and lines:
        lines = lines[:-1]
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
    out_dir = "proofs_dump_repo"
    n = dump_new(cfg.results_jsonl, out_dir)
    print(f"Добавлено новых файлов: {n}")
