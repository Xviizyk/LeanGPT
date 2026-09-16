"""
Обёртка над Lean REPL (leanprover-community/repl).

Поддерживает два режима:
1. verify(code) — проверка целого куска кода целиком (как раньше).
2. Пошаговый режим по тактикам через proofState: open_goal() -> run_tactic()
   -> получаем новое состояние (goals) после каждой тактики, а не только
   в самом конце. Это даёт частичный сигнал ("эта тактика продвинула
   доказательство") вместо бинарного "весь текст скомпилировался / нет".

Установка REPL:
    git clone https://github.com/leanprover-community/repl
    cd repl && lake build
    # бинарник: .lake/build/bin/repl
"""

from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VerifyResult:
    ok: bool
    has_sorry: bool
    errors: list[str]
    warnings: list[str]
    raw: dict


@dataclass
class TacticState:
    proof_state_id: int
    goals: list[str]           # человекочитаемые оставшиеся цели
    done: bool                 # goals пуст -> доказательство завершено
    error: Optional[str] = None


class LeanReplError(RuntimeError):
    pass


class LeanRepl:
    def __init__(
        self,
        repl_bin: str,
        project_dir: str,
        timeout_sec: float = 60.0,
        env_file: Optional[str] = None,
    ) -> None:
        self.repl_bin = repl_bin
        self.project_dir = project_dir
        self.timeout_sec = timeout_sec
        self.env_file = env_file
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._env_id: Optional[int] = None

    def __enter__(self) -> "LeanRepl":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        self._proc = subprocess.Popen(
            [self.repl_bin],
            cwd=self.project_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if self.env_file:
            self._env_id = self._pick_env(self.env_file)

    def stop(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            self._proc = None

    def restart(self) -> None:
        self.stop()
        self.start()

    def _send(self, payload: dict) -> dict:
        assert self._proc is not None, "REPL не запущен, вызови start()"
        line = json.dumps(payload)
        with self._lock:
            self._proc.stdin.write(line + "\n\n")
            self._proc.stdin.flush()
            out_lines = []
            while True:
                out_line = self._proc.stdout.readline()
                if out_line == "":
                    raise LeanReplError("REPL завершился неожиданно")
                if out_line.strip() == "":
                    break
                out_lines.append(out_line)
        raw = "".join(out_lines).strip()
        if not raw:
            raise LeanReplError("Пустой ответ от REPL")
        return json.loads(raw)

    def _pick_env(self, import_module: str) -> int:
        resp = self._send({"cmd": f"import {import_module}"})
        env = resp.get("env")
        if env is None:
            raise LeanReplError(f"Не удалось создать окружение: {resp}")
        return env

    # -- режим 1: весь код целиком ---------------------------------------

    def verify(self, code: str) -> VerifyResult:
        payload = {"cmd": code}
        if self._env_id is not None:
            payload["env"] = self._env_id
        try:
            resp = self._send(payload)
        except LeanReplError:
            self.restart()
            return VerifyResult(ok=False, has_sorry=False, errors=["repl_crashed"], warnings=[], raw={})

        messages = resp.get("messages", [])
        errors = [m["data"] for m in messages if m.get("severity") == "error"]
        warnings = [m["data"] for m in messages if m.get("severity") == "warning"]
        has_sorry = any("sorry" in w.lower() for w in warnings) or "sorry" in code

        return VerifyResult(ok=(len(errors) == 0), has_sorry=has_sorry, errors=errors, warnings=warnings, raw=resp)

    # -- режим 2: пошагово по тактикам ------------------------------------

    def open_goal(self, statement: str) -> TacticState:
        """Открыть новую цель из формулировки теоремы (statement должен
        заканчиваться на ':= by' — REPL вернёт исходный proofState с целями)."""
        code = statement if statement.rstrip().endswith("by") else f"{statement} := by"
        payload = {"cmd": code}
        if self._env_id is not None:
            payload["env"] = self._env_id
        try:
            resp = self._send(payload)
        except LeanReplError:
            self.restart()
            return TacticState(proof_state_id=-1, goals=[], done=False, error="repl_crashed")

        errors = [m["data"] for m in resp.get("messages", []) if m.get("severity") == "error"]
        if errors:
            return TacticState(proof_state_id=-1, goals=[], done=False, error="; ".join(errors))

        sorries = resp.get("sorries", [])
        if not sorries:
            # либо доказательство пустое и сразу решено (rfl-подобная теорема), либо ошибка формата
            return TacticState(proof_state_id=resp.get("env", -1), goals=[], done=True)

        goal = sorries[0]
        return TacticState(
            proof_state_id=goal["proofState"],
            goals=[goal.get("goal", "")],
            done=False,
        )

    def run_tactic(self, state: TacticState, tactic: str) -> TacticState:
        """Применить одну тактику к текущему proofState, получить новое состояние."""
        if state.done or state.error:
            return state

        payload = {"tactic": tactic, "proofState": state.proof_state_id}
        try:
            resp = self._send(payload)
        except LeanReplError:
            self.restart()
            return TacticState(proof_state_id=-1, goals=state.goals, done=False, error="repl_crashed")

        if "error" in resp or resp.get("messages") and any(
            m.get("severity") == "error" for m in resp.get("messages", [])
        ):
            errs = resp.get("error") or [
                m["data"] for m in resp.get("messages", []) if m.get("severity") == "error"
            ]
            return TacticState(proof_state_id=state.proof_state_id, goals=state.goals, done=False, error=str(errs))

        new_goals = resp.get("goals", [])
        return TacticState(
            proof_state_id=resp.get("proofState", state.proof_state_id),
            goals=new_goals,
            done=(len(new_goals) == 0),
        )
