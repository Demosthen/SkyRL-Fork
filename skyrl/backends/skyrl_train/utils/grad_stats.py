"""Worker-side gradient snapshots and inner products, for gradient diagnostics.

A *slot* is a snapshot of a model's accumulated gradient: a list with one entry per
trainable parameter (the local shard under FSDP2, ``None`` if the parameter had no
gradient). ``gram_matrix`` returns all pairwise inner products between slots. It
computes them shard-locally in float64, then all-reduces along the mesh dimensions
where the gradients are sharded (replicated dimensions are not summed). Only a
K x K matrix leaves the workers.
"""

from typing import Iterable, Optional

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

GradSlot = list[Optional[torch.Tensor]]


def snapshot_grads(params: Iterable[torch.nn.Parameter]) -> GradSlot:
    """Copy the current ``.grad`` of each parameter (None stays None)."""
    return [None if p.grad is None else p.grad.detach().clone() for p in params]


def _local_and_shard_groups(t: torch.Tensor) -> tuple[torch.Tensor, tuple]:
    """Local tensor, and the process groups to sum over to get a global inner product."""
    if not isinstance(t, DTensor):
        return t, ()
    groups = []
    for dim, placement in enumerate(t.placements):
        if placement.is_shard():
            groups.append(t.device_mesh.get_group(dim))
        elif placement.is_partial():
            raise ValueError("gradients with Partial placement are not supported")
    return t.to_local(), tuple(groups)


def gram_matrix(slots: list[GradSlot]) -> torch.Tensor:
    """G[i, j] = <slot_i, slot_j> over all parameters, as a float64 CPU tensor.

    Collective: every rank holding a shard must call this with the same slots.
    """
    k = len(slots)
    n = len(slots[0]) if slots else 0
    if any(len(s) != n for s in slots):
        raise ValueError("slots were taken from different parameter lists")
    partial_by_groups: dict[tuple, torch.Tensor] = {}
    for i in range(n):
        entries = [s[i] for s in slots]
        ref = next((e for e in entries if e is not None), None)
        if ref is None:
            continue
        ref_local, groups = _local_and_shard_groups(ref)
        rows = [
            (
                torch.zeros(ref_local.numel(), dtype=torch.float64, device=ref_local.device)
                if e is None
                else _local_and_shard_groups(e)[0].reshape(-1).to(torch.float64)
            )
            for e in entries
        ]
        m = torch.stack(rows)
        acc = partial_by_groups.get(groups)
        partial_by_groups[groups] = m @ m.T if acc is None else acc + m @ m.T
    gram = torch.zeros(k, k, dtype=torch.float64)
    for groups, part in partial_by_groups.items():
        for group in groups:
            dist.all_reduce(part, op=dist.ReduceOp.SUM, group=group)
        gram += part.cpu()
    return gram
