from __future__ import annotations

import queue
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

from .repl import LeanRepl


@dataclass
class FailedResult:
    
    ok: bool = False
    has_sorry: bool = False
    errors: list = field(default_factory=list)


class ReplPool:

    def __init__(
        self,
        repl_bin: str,
        project_dir: str,
        pool_size: int = 4,
        timeout_sec: float = 60.0,
        env_file: Optional[str] = None,
    ) -> None:
        self.repl_bin = repl_bin
        self.project_dir = project_dir
        self.pool_size = pool_size
        self.timeout_sec = timeout_sec
        self.env_file = env_file
        self._pool: "queue.Queue[LeanRepl]" = queue.Queue()
        self._all: list[LeanRepl] = []

    def start(self) -> None:
        for _ in range(self.pool_size):
            repl = LeanRepl(
                self.repl_bin, self.project_dir, self.timeout_sec, self.env_file
            )
            repl.start()
            self._all.append(repl)
            self._pool.put(repl)

    def stop(self) -> None:
        for repl in self._all:
            repl.stop()
        self._all.clear()

    def __enter__(self) -> "ReplPool":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    @contextmanager
    def checkout(self):
        repl = self._pool.get()
        try:
            yield repl
        finally:
            self._pool.put(repl)

    def verify_many(self, codes: list[str]) -> list:
        results: list = [None] * len(codes)
        lock = threading.Lock()

        def worker(i: int, code: str) -> None:
            with self.checkout() as repl:
                try:
                    result = repl.verify(code)
                except Exception as e:
                    result = FailedResult(errors=[f"repl error: {e!r}"])
                    try:
                        repl.stop()
                        repl.start()
                    except Exception:
                        pass
            with lock:
                results[i] = result

        threads = [
            threading.Thread(target=worker, args=(i, c)) for i, c in enumerate(codes)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return [r if r is not None else FailedResult(errors=["no result"]) for r in results]