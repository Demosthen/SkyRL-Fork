"""CPU test for the trainable-only (LoRA) checkpoint subset helper.

uv run --isolated --extra dev pytest tests/backends/skyrl_train/test_trainable_checkpoint.py
"""

import pytest
import torch

from skyrl.backends.skyrl_train.distributed.fsdp_strategy import _trainable_subset


def _model():
    m = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    m[0].requires_grad_(False)  # frozen "base" layer
    return m


def test_subset_keeps_only_trainable():
    m = _model()
    sub = _trainable_subset(m, m.state_dict())
    assert set(sub) == {"1.weight", "1.bias"}


def test_subset_roundtrip_restores_trainable_and_keeps_frozen():
    m = _model()
    saved = {k: v.clone() for k, v in _trainable_subset(m, m.state_dict()).items()}
    frozen = m[0].weight.detach().clone()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0) if p.requires_grad else None
    m.load_state_dict(saved, strict=False)
    torch.testing.assert_close(m[1].weight, saved["1.weight"])
    torch.testing.assert_close(m[0].weight, frozen)


def test_subset_raises_when_keys_do_not_cover_trainable():
    m = _model()
    with pytest.raises(KeyError):
        _trainable_subset(m, {"0.weight": m[0].weight})
