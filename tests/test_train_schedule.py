from leangpt.schedules import lr_schedule


def test_warmup_increases_lr():
    lr_start = lr_schedule(0, total_steps=100, base_lr=1.0, warmup_steps=10)
    lr_mid_warmup = lr_schedule(5, total_steps=100, base_lr=1.0, warmup_steps=10)
    assert lr_start < lr_mid_warmup


def test_reaches_base_lr_at_end_of_warmup():
    lr = lr_schedule(9, total_steps=100, base_lr=1.0, warmup_steps=10)
    assert abs(lr - 1.0) < 1e-06


def test_decays_after_warmup():
    lr_after_warmup = lr_schedule(10, total_steps=100, base_lr=1.0, warmup_steps=10)
    lr_near_end = lr_schedule(95, total_steps=100, base_lr=1.0, warmup_steps=10)
    assert lr_near_end < lr_after_warmup


def test_never_goes_below_min_ratio():
    lr_final = lr_schedule(
        100, total_steps=100, base_lr=1.0, warmup_steps=10, min_lr_ratio=0.1
    )
    assert lr_final >= 0.1 - 1e-06
