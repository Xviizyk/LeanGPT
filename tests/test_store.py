import json

import pytest

from lean4gen.store import JsonlStore, code_hash, normalize_code, tactic_signature


def test_normalize_code_collapses_whitespace():
    a = "theorem foo : 1 = 1 := by\n  rfl"
    b = "theorem foo : 1 = 1 := by rfl"
    assert normalize_code(a) == normalize_code(b)


def test_code_hash_stable_under_whitespace_changes():
    a = "theorem foo := by decide"
    b = "theorem   foo   :=   by   decide"
    assert code_hash(a) == code_hash(b)


def test_tactic_signature_extracts_known_tactics():
    code = "theorem foo := by simp; ring"
    sig = tactic_signature(code)
    assert "simp" in sig and "ring" in sig


def test_tactic_signature_fallback_to_other():
    code = "theorem foo := by exact_this_is_not_a_known_tactic"
    assert tactic_signature(code) == "other"


def test_add_rejects_exact_duplicate_successful_proof(tmp_path):
    store = JsonlStore(str(tmp_path / "results.jsonl"))
    added_1 = store.add("theorem foo : 1 = 1", "theorem foo : 1 = 1 := by rfl", ok=True, has_sorry=False, errors=[])
    added_2 = store.add("theorem foo : 1 = 1", "theorem foo : 1 = 1 := by rfl", ok=True, has_sorry=False, errors=[])
    assert added_1 is True
    assert added_2 is False


def test_add_keeps_different_proofs_of_same_theorem(tmp_path):
    store = JsonlStore(str(tmp_path / "results.jsonl"))
    store.add("theorem foo : 1 = 1", "theorem foo : 1 = 1 := by rfl", ok=True, has_sorry=False, errors=[])
    added = store.add("theorem foo : 1 = 1", "theorem foo : 1 = 1 := by decide", ok=True, has_sorry=False, errors=[])
    assert added is True  # разный путь к той же теореме — не дубликат

    proofs = list(store.iter_successful())
    assert len(proofs) == 2


def test_add_allows_duplicate_failures(tmp_path):
    store = JsonlStore(str(tmp_path / "results.jsonl"))
    store.add("theorem foo", "theorem foo := by bogus_tactic", ok=False, has_sorry=False, errors=["error"])
    added_2 = store.add("theorem foo", "theorem foo := by bogus_tactic", ok=False, has_sorry=False, errors=["error"])
    assert added_2 is True  # неуспешные попытки не дедуплицируются — тоже сигнал


def test_iter_balanced_caps_per_signature(tmp_path):
    store = JsonlStore(str(tmp_path / "results.jsonl"))
    for i in range(10):
        store.add(f"theorem t{i}", f"theorem t{i} := by decide -- variant {i}", ok=True, has_sorry=False, errors=[])

    balanced = list(store.iter_balanced(max_per_signature=3, max_per_statement=100))
    assert len(balanced) == 3


def test_iter_balanced_caps_per_statement(tmp_path):
    store = JsonlStore(str(tmp_path / "results.jsonl"))
    for i in range(10):
        store.add("theorem same_thm", f"theorem same_thm := by simp -- v{i}", ok=True, has_sorry=False, errors=[])

    balanced = list(store.iter_balanced(max_per_signature=1000, max_per_statement=2))
    assert len(balanced) == 2


def test_stats_reports_success_rate(tmp_path):
    store = JsonlStore(str(tmp_path / "results.jsonl"))
    store.add("theorem a", "theorem a := by rfl", ok=True, has_sorry=False, errors=[])
    store.add("theorem b", "theorem b := by bogus", ok=False, has_sorry=False, errors=["e"])

    stats = store.stats()
    assert stats["total"] == 2
    assert stats["ok"] == 1
    assert stats["success_rate"] == pytest.approx(0.5)
