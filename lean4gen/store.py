"""
JSONL-хранилище результатов + дедупликация/балансировка успешных
примеров, чтобы self-play датасет не схлопывался в повторение одних
и тех же тривиальных доказательств (by decide / by rfl и т.п.).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Record:
    statement: str
    code: str
    ok: bool
    has_sorry: bool
    errors: list[str]
    timestamp: float


def normalize_code(code: str) -> str:
    """Убрать пробелы/переносы для сравнения "по сути одинаковых" доказательств."""
    return re.sub(r"\s+", " ", code).strip()


def code_hash(code: str) -> str:
    return hashlib.sha256(normalize_code(code).encode("utf-8")).hexdigest()


def tactic_signature(code: str) -> str:
    """Грубая сигнатура "какими тактиками решено" — для балансировки по типам
    (чтобы не переполнить датасет однотипными rfl/decide-доказательствами)."""
    tactics = re.findall(r"\b(rfl|decide|simp|ring|omega|induction|cases|exact|apply|linarith)\b", code)
    return ",".join(sorted(set(tactics))) or "other"


class JsonlStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seen_hashes: set[str] | None = None

    def add(self, statement: str, code: str, ok: bool, has_sorry: bool, errors: list[str]) -> bool:
        """Вернёт False, если запись — дубликат уже сохранённого успешного примера
        (дубликаты неуспешных попыток всё равно пишем — они тоже полезный сигнал)."""
        if ok and self._is_duplicate(code):
            return False

        rec = Record(statement=statement, code=code, ok=ok, has_sorry=has_sorry, errors=errors, timestamp=time.time())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

        if ok:
            self._register_hash(code)
        return True

    def _load_hashes(self) -> set[str]:
        if self._seen_hashes is not None:
            return self._seen_hashes
        hashes: set[str] = set()
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    if rec["ok"]:
                        hashes.add(code_hash(rec["code"]))
        self._seen_hashes = hashes
        return hashes

    def _is_duplicate(self, code: str) -> bool:
        return code_hash(code) in self._load_hashes()

    def _register_hash(self, code: str) -> None:
        self._load_hashes().add(code_hash(code))

    def iter_successful(self, exclude_sorry: bool = True):
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["ok"] and (not exclude_sorry or not rec["has_sorry"]):
                    yield rec

    def iter_balanced(self, exclude_sorry: bool = True, max_per_signature: int = 200, max_per_statement: int = 20):
        """Итератор по успешным примерам с двумя ограничениями:
        - max_per_signature: не более N примеров на "сигнатуру тактик" в целом
          (не даёт датасету зарасти тысячей одинаковых `by decide`).
        - max_per_statement: не более N *разных* доказательств одной и той же
          теоремы (разные пути к одному результату — полезный сигнал, но не
          нужно 200 вариаций для одной тривиальной леммы).
        Дубликаты по коду (ровно то же доказательство) отсекаются ещё на
        этапе add(), сюда не попадают."""
        sig_counts: Counter[str] = Counter()
        per_statement: Counter[str] = Counter()
        for rec in self.iter_successful(exclude_sorry=exclude_sorry):
            sig = tactic_signature(rec["code"])
            stmt_key = normalize_code(rec["statement"])
            if sig_counts[sig] >= max_per_signature:
                continue
            if per_statement[stmt_key] >= max_per_statement:
                continue
            sig_counts[sig] += 1
            per_statement[stmt_key] += 1
            yield rec

    def stats(self) -> dict:
        if not self.path.exists():
            return {}
        total, ok, sorry = 0, 0, 0
        sig_counts: Counter[str] = Counter()
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                total += 1
                if rec["ok"]:
                    ok += 1
                    sig_counts[tactic_signature(rec["code"])] += 1
                if rec["has_sorry"]:
                    sorry += 1
        return {"total": total, "ok": ok, "success_rate": ok / total if total else 0.0,
                "with_sorry": sorry, "by_tactic_signature": dict(sig_counts)}
