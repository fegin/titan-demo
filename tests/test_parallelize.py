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
from torch.distributed.tensor import DTensor


@pytest.fixture(autouse=True)
def _reset_distributed():
    """Tear down torch.distributed between tests so each test sets its own world_size.

    _setup_fake_distributed is idempotent on ``is_initialized()``, so without
    this fixture a later test would silently reuse the earlier world_size.
    """
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def _build():
    spec = make_model_spec("debugmodel", seq_len=128)
    model, fake_mode = build_fake_model(spec.model, dtype=torch.bfloat16)
    return spec, model, fake_mode


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
    spec, model, _ = _build()
    pd = make_parallel_dims(world_size=4)  # dp_shard=4, tp=1
    parallelize_fake_model(model, spec=spec, parallel_dims=pd)

    params = list(model.parameters())
    assert all(isinstance(p, DTensor) for p in params)


def test_tp_only_shards_attention_and_ffn():
    spec, model, _ = _build()
    pd = make_parallel_dims(world_size=2, tp=2)
    parallelize_fake_model(model, spec=spec, parallel_dims=pd)

    # wq is colwise -> Shard(0) on the TP axis.
    wq = model.layers["0"].attention.qkv_linear.wq.weight
    assert isinstance(wq, DTensor)
    assert any(p.is_shard() for p in wq.placements)


def test_tp_plus_fsdp_2d():
    spec, model, _ = _build()
    pd = make_parallel_dims(world_size=8, tp=2)  # dp_shard=4
    parallelize_fake_model(model, spec=spec, parallel_dims=pd)

    params = list(model.parameters())
    assert all(isinstance(p, DTensor) for p in params)
    # Multi-axis mesh: at least one param should live on a 2-axis mesh.
    assert any(p.device_mesh.ndim >= 2 for p in params)
