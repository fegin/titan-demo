"""Smoke tests for ``titan_demo.parallelize_fake_model``."""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from titan_demo import (
    build_fake_model,
    make_model_spec,
    make_parallel_dims,
    parallelize_fake_model,
)
from titan_demo.parallelize import _setup_fake_distributed
from torch.distributed.tensor import DTensor


@pytest.fixture(autouse=True)
def _reset_distributed():
    """Tear down torch.distributed between tests so each test sets its own world_size.

    _setup_fake_distributed raises on a world_size mismatch when the process
    group is already initialized; this fixture ensures each test starts from
    a clean slate and can pick any world_size it needs.
    """
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def _build():
    spec = make_model_spec("debugmodel", seq_len=128)
    model, fake_mode = build_fake_model(spec.model, dtype=torch.bfloat16)
    return model, fake_mode


def test_make_parallel_dims_fills_leftover_dp_shard():
    pd = make_parallel_dims(world_size=8, tp=2)
    assert pd.dp_shard == 4
    assert pd.tp == 2
    assert pd.full_dtensor is True


def test_make_parallel_dims_validates_product():
    with pytest.raises(AssertionError):
        # 2 * 1 * 1 * 2 * 1 = 4, not 8
        make_parallel_dims(world_size=8, dp_replicate=2, dp_shard=1, tp=2)


def test_fsdp_only_shards_params_to_dtensor():
    model, _ = _build()
    pd = make_parallel_dims(world_size=4)  # dp_shard=4, tp=1
    parallelize_fake_model(model, parallel_dims=pd)

    params = list(model.parameters())
    assert all(isinstance(p, DTensor) for p in params)


def test_tp_only_shards_attention_and_ffn():
    model, _ = _build()
    pd = make_parallel_dims(world_size=2, tp=2)
    parallelize_fake_model(model, parallel_dims=pd)

    # wq is colwise -> Shard(0) on the TP axis.
    wq = model.layers["0"].attention.qkv_linear.wq.weight
    assert isinstance(wq, DTensor)
    assert any(p.is_shard() for p in wq.placements)


def test_setup_fake_distributed_rejects_world_size_mismatch():
    """Re-init with a different world_size must raise, not silently reuse."""
    _setup_fake_distributed(world_size=4)
    # Calling again with the same size is a no-op.
    _setup_fake_distributed(world_size=4)
    # A different size on the existing PG must raise with a clear message.
    with pytest.raises(RuntimeError, match=r"world_size=4.*world_size=8"):
        _setup_fake_distributed(world_size=8)
