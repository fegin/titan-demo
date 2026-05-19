"""Smoke tests for the titan_demo public API."""

from __future__ import annotations

import torch

from titan_demo import build_fake_model, make_model_spec
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
    assert all(p.device.type == "meta" for p in params)
    assert all(p.dtype == torch.bfloat16 for p in params)

    # debugmodel has ~6M params; just check it is in the expected ballpark.
    total = sum(p.numel() for p in params)
    assert 1_000_000 < total < 50_000_000

    assert fake_mode is not None


def test_user_can_modify_config_before_build() -> None:
    """The split API lets users edit sharding/config between the two steps."""
    spec = make_model_spec("debugmodel", seq_len=128)

    # make_model_spec already installed defaults via
    # set_llama3_sharding_config; verify the sharding_config slot is
    # reachable, then overwrite it with a custom plan to simulate a user
    # tweaking the config between make_model_spec and build_fake_model.
    layer = spec.model.layers[0]
    assert hasattr(layer.attention, "sharding_config")

    from torchtitan.models.llama3.sharding import set_llama3_sharding_config

    set_llama3_sharding_config(spec.model, loss_parallel=True, enable_sp=True)
    assert spec.model.layers[0].attention.sharding_config is not None

    # Build should succeed against the modified config.
    model, _ = build_fake_model(spec.model)
    assert sum(p.numel() for p in model.parameters()) > 0
