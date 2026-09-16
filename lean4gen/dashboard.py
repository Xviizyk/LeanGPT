"""
Живое консольное отображение процесса генерации — в духе TUI у LLM-клиентов
(токены/сек, текущий промпт, статистика), но своё, без форка стороннего
клиента: используется только библиотека `rich` (Live + Table).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from rich.console import Console
from rich.live import Live
from rich.table import Table


@dataclass
class DashboardState:
    level: int = 0
    current_statement: str = ""
    total_candidates: int = 0
    total_proved: int = 0
    total_attempts: int = 0
    tps_window: deque = field(default_factory=lambda: deque(maxlen=20))  # (n_tokens, dt) пар
    started_at: float = field(default_factory=time.time)


class Dashboard:
    def __init__(self) -> None:
        self.state = DashboardState()
        self._console = Console()
        self._live: Live | None = None

    def __enter__(self) -> "Dashboard":
        self._live = Live(self._render(), console=self._console, refresh_per_second=4)
        self._live.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self._live:
            self._live.__exit__(*exc)

    def record_generation(self, n_tokens: int, elapsed_sec: float) -> None:
        """Вызывается после каждого generate() — для расчёта скользящего TPS."""
        if elapsed_sec > 0:
            self.state.tps_window.append((n_tokens, elapsed_sec))
        self.state.total_candidates += 1
        self._refresh()

    def record_attempt(self, statement: str, level: int, ok: bool) -> None:
        self.state.current_statement = statement
        self.state.level = level
        self.state.total_attempts += 1
        if ok:
            self.state.total_proved += 1
        self._refresh()

    def _tps(self) -> float:
        if not self.state.tps_window:
            return 0.0
        total_tokens = sum(t for t, _ in self.state.tps_window)
        total_time = sum(d for _, d in self.state.tps_window)
        return total_tokens / total_time if total_time > 0 else 0.0

    def _render(self) -> Table:
        elapsed = time.time() - self.state.started_at
        success_rate = (
            self.state.total_proved / self.state.total_attempts if self.state.total_attempts else 0.0
        )

        table = Table(title="LeanGPT — self-play", show_header=False)
        table.add_row("Curriculum level", str(self.state.level))
        table.add_row("Текущая теорема", self.state.current_statement[:70])
        table.add_row("Токенов/сек (скользящее)", f"{self._tps():.1f}")
        table.add_row("Сгенерировано кандидатов", str(self.state.total_candidates))
        table.add_row("Попыток / доказано", f"{self.state.total_attempts} / {self.state.total_proved}")
        table.add_row("Success rate", f"{success_rate:.1%}")
        table.add_row("Время сессии", f"{elapsed:.0f}s")
        return table

    def _refresh(self) -> None:
        if self._live:
            self._live.update(self._render())
