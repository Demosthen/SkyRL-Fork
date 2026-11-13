"""CPU tests for skyrl.backends.skyrl_train.utils.grad_stats.

uv run --isolated --extra dev pytest tests/backends/skyrl_train/test_grad_stats.py
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from skyrl.backends.skyrl_train.utils.grad_stats import gram_matrix, snapshot_grads

K, SHAPES = 3, [(7, 5), (4,), (1, 3)]  # 7 rows: uneven Shard(0) over 2 ranks


def _full_vectors(seed: int = 0) -> list[list[torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    return [[torch.randn(s, generator=g, dtype=torch.float64) for s in SHAPES] for _ in range(K)]


def _expected(vectors) -> torch.Tensor:
    flat = torch.stack([torch.cat([t.reshape(-1) for t in v]) for v in vectors])
    return flat @ flat.T


def test_gram_single_process_matches_dense():
    vectors = _full_vectors()
    vectors[2][1] = None  # missing grad counts as zero
    dense = [[torch.zeros(s, dtype=torch.float64) if t is None else t for s, t in zip(SHAPES, v)] for v in vectors]
    torch.testing.assert_close(gram_matrix(vectors), _expected(dense))


def test_snapshot_copies_and_keeps_none():
    a, b = torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(2))
    a.grad = torch.full((2,), 3.0)
    snap = snapshot_grads([a, b])
    a.grad.zero_()
    assert snap[1] is None and torch.equal(snap[0], torch.full((2,), 3.0))


def _worker(rank: int, world: int, port: int, result_q):
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor
    from torch.distributed.device_mesh import init_device_mesh

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    mesh = init_device_mesh("cpu", (world,))
    vectors = _full_vectors()
    sharded = [[distribute_tensor(t, mesh, [Shard(0)]) for t in v] for v in vectors]
    replicated = [[distribute_tensor(t, mesh, [Replicate()]) for t in v] for v in vectors]
    # numpy, not torch: torch tensors cross a Queue via shared-memory fds that die with this process
    result_q.put((rank, gram_matrix(sharded).numpy(), gram_matrix(replicated).numpy()))
    dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("world", [2])
def test_gram_dtensor_sharded_and_replicated(world):
    ctx = mp.get_context("spawn")
    q, port = ctx.Queue(), _free_port()
    procs = [ctx.Process(target=_worker, args=(r, world, port, q)) for r in range(world)]
    for p in procs:
        p.start()
    results = [q.get(timeout=120) for _ in procs]
    for p in procs:
        p.join(timeout=60)
    expected = _expected(_full_vectors())
    for _, g_sharded, g_replicated in results:
        torch.testing.assert_close(torch.from_numpy(g_sharded), expected)
        torch.testing.assert_close(torch.from_numpy(g_replicated), expected)  # replicated dims are not summed
