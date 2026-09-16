import pytest

try:
    import torch
except Exception:
    torch = None

if torch is None:
    pytest.skip("torch not available", allow_module_level=True)

from leangpt.device import pick_device


def test_pick_device_returns_valid_value():
    assert pick_device() in ("cuda", "mps", "cpu")
