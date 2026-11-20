"""CPU test: Worker.set_optimizer_hparams overrides AdamW hyperparameters, including ones
restored by optimizer.load_state_dict from a checkpoint."""

from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.workers.worker import Worker


def _adamw(**kwargs):
    return torch.optim.AdamW([torch.nn.Parameter(torch.zeros(3))], lr=1e-3, **kwargs)


def test_set_optimizer_hparams_overrides_checkpoint_values():
    saved = _adamw(betas=(0.9, 0.999), weight_decay=1e-2).state_dict()
    worker = SimpleNamespace(optimizer=_adamw())
    worker.optimizer.load_state_dict(saved)  # restores betas/weight_decay from the checkpoint
    assert worker.optimizer.param_groups[0]["betas"] == (0.9, 0.999)

    Worker.set_optimizer_hparams(worker, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    group = worker.optimizer.param_groups[0]
    assert (group["betas"], group["eps"], group["weight_decay"]) == ((0.9, 0.95), 1e-8, 0.0)


def test_set_optimizer_hparams_rejects_unknown_key_and_skips_without_optimizer():
    with pytest.raises(KeyError):
        Worker.set_optimizer_hparams(SimpleNamespace(optimizer=_adamw()), momentum=0.9)
    Worker.set_optimizer_hparams(SimpleNamespace(optimizer=None), betas=(0.9, 0.95))
