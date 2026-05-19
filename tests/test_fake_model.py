"""Smoke tests for the titan_demo public API."""

from __future__ import annotations

import torch

from titan_demo import build_fake_model, make_model_spec, run_forward
from torch._subclasses.fake_tensor import FakeTensor


def test_make_model_spec_returns_unbuilt_config() -> None:
    spec = make_model_spec("debugmodel", seq_len=128)

    assert spec.name == "llama3"
    assert spec.flavor == "debugmodel"
    assert spec.model.rope.max_seq_len == 128

    # The config has not been built; it's still a dataclass-like config.
    assert not isinstance(spec.model, torch.nn.Module)


def test_build_fake_model_returns_fake_params() -> None:
    spec = make_model_spec("debugmodel", seq_len=128)
    model, fake_mode = build_fake_model(spec.model, dtype=torch.bfloat16)

    params = list(model.parameters())
    assert len(params) > 0
    assert all(isinstance(p, FakeTensor) for p in params)
    assert all(p.device.type == "cpu" for p in params)
    assert all(p.dtype == torch.bfloat16 for p in params)

    # debugmodel has ~6M params; just check it is in the expected ballpark.
    total = sum(p.numel() for p in params)
    assert 1_000_000 < total < 50_000_000

    assert fake_mode is not None


def test_user_can_modify_config_before_build() -> None:
    """The split API lets users edit sharding/config between the two steps."""
    spec = make_model_spec("debugmodel", seq_len=128)

    # Sanity-check a representative sharding slot is reachable for modification
    # (set_llama3_sharding_config has not been called yet, so it is the
    # default left by model_registry -- which may be None).
    layer = spec.model.layers[0]
    assert hasattr(layer.attention, "sharding_config")

    # Pretend the user installed a custom sharding plan.
    from torchtitan.models.llama3.sharding import set_llama3_sharding_config

    set_llama3_sharding_config(spec.model, loss_parallel=True, enable_sp=True)
    assert spec.model.layers[0].attention.sharding_config is not None

    # Build should succeed against the modified config.
    model, _ = build_fake_model(spec.model)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_run_forward_returns_expected_logits_shape() -> None:
    spec = make_model_spec("debugmodel", seq_len=128)
    model, fake_mode = build_fake_model(spec.model, dtype=torch.bfloat16)

    batch_size = 2
    seq_len = 64
    logits = run_forward(model, fake_mode, batch_size=batch_size, seq_len=seq_len)

    assert isinstance(logits, FakeTensor)
    # debugmodel vocab_size = 2048
    assert tuple(logits.shape) == (batch_size, seq_len, 2048)
    assert logits.dtype == torch.bfloat16
    assert logits.device.type == "cpu"
