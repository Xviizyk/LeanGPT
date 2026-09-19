from __future__ import annotations
import json
import re
import time
from pathlib import Path


class GenDump:
    """Сохраняет каждого сгенерированного кандидата в .lean-файл."""

    def __init__(self, gen, out_dir: str | Path) -> None:
        self.gen = gen
        self.root = Path(out_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "index.jsonl"
        self.level = 0
        self.call = 0
        self._orig = gen.generate
        gen.generate = self._wrapped

    def _wrapped(self, prompt: str) -> list[str]:
        cands = self._orig(prompt)
        self._save(prompt, cands)
        return cands

    def _save(self, prompt: str, cands: list[str]) -> None:
        self.call += 1
        name = re.search(r"theorem\s+(\S+)", prompt)
        slug = re.sub(r"[^\w\-]", "_", name.group(1)) if name else "stmt"
        d = self.root / f"level{self.level}"
        d.mkdir(parents=True, exist_ok=True)
        with self.index.open("a", encoding="utf-8") as idx:
            for i, completion in enumerate(cands):
                path = d / f"{self.call:06d}_{slug}_c{i}.lean"
                path.write_text(prompt + completion + "\n", encoding="utf-8")
                idx.write(
                    json.dumps(
                        {
                            "ts": time.time(),
                            "call": self.call,
                            "level": self.level,
                            "candidate": i,
                            "file": str(path),
                            "prompt": prompt,
                            "completion": completion,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )