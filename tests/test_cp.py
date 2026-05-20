"""Smoke test for ``cp > 1`` under FakeTensorMode.

This test exercises every patch in ``titan_demo/_patches.py`` end-to-end:
HOO dispatch in ``FSDPMemTracker``, the ``ModTracker`` ``GraphModule`` skip,
``_StridedShard.local_shard_size_and_offset`` ``.tolist()`` bypass, the
eager-``flex_attention`` dispatch in TorchTitan's ``FlexAttention``, and
the ``redistribute_cost`` short-circuit that avoids the Dijkstra blow-up
on ``_StridedShard`` placements. If any of those patches stops working
after a PyTorch or TorchTitan upgrade, this test should fail.

Lives in its own file (not ``test_memory.py``) because PyTorch's FSDP
mesh caches survive ``destroy_process_group``, so we only run one full
``estimate_memory`` per file.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from titan_demo import (
    build_fake_model,
    estimate_memory,
    make_model_spec,
    make_parallel_dims,
    parallelize_fake_model,
)


@pytest.fixture(autouse=True)
def _reset_distributed():
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def test_estimate_memory_with_cp() -> None:
    # tp=2, cp=2, world_size=4 -> dp_shard=1. seq_len must be divisible
    # by tp * 2 * cp = 8; 128 satisfies that and equals rope.max_seq_len.
    spec = make_model_spec("debugmodel", seq_len=128, attn_backend="flex")
    model, fake_mode = build_fake_model(spec.model, dtype=torch.bfloat16)
    pd = make_parallel_dims(world_size=4, tp=2, cp=2)
    parallelize_fake_model(model, parallel_dims=pd)

    snap = estimate_memory(model, fake_mode, pd, batch_size=1, seq_len=128)

    assert any(b.get("Total", 0) > 0 for b in snap.values())
