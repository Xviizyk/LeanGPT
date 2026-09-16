from lean4gen.curriculum import (
    curriculum_batch,
    eval_batch,
    level_0,
    level_3,
    should_advance,
)


def test_level_0_is_deterministic():
    a = level_0(5, seed_offset=0)
    b = level_0(5, seed_offset=0)
    assert a == b


def test_level_0_eval_offset_does_not_overlap_train():
    train = set(level_0(20, seed_offset=0))
    eval_ = set(level_0(20, seed_offset=10000))
    assert train.isdisjoint(eval_)


def test_level_3_eval_offset_does_not_overlap_train():
    train = set(level_3(20, seed_offset=0))
    eval_ = set(level_3(20, seed_offset=10000))
    assert train.isdisjoint(eval_)


def test_curriculum_batch_includes_lower_levels():
    batch = curriculum_batch(current_level=2, n_per_level=8)
    assert any(("const_add" in s for s in batch))
    assert any(
        (
            "add_comm" in s or "add_zero" in s or "zero_add" in s or ("mul_one" in s)
            for s in batch
        )
    )
    assert any(("and_comm" in s or "or_comm" in s or "not_not" in s for s in batch))


def test_eval_batch_returns_statements():
    batch = eval_batch(current_level=0, n=5)
    assert len(batch) == 5
    assert all(("theorem" in s for s in batch))


def test_should_advance_threshold():
    assert should_advance(0.6) is True
    assert should_advance(0.59) is False
    assert should_advance(1.0) is True
