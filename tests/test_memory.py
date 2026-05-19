"""Smoke tests for ``titan_demo.estimate_memory``.

A note on state isolation: each call to ``parallelize_fake_model``
initializes a fake process group and builds device meshes via
torch.distributed APIs. Running the full estimate pipeline more than
once in the same Python process can leave behind cached mesh state
that confuses FSDP on the next run (it asserts mesh-object identity).
We therefore run ``estimate_memory`` at most once per test, and verify
the formatter with a hand-built snapshot.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from titan_demo import (
    build_fake_model,
    estimate_memory,
    format_memory_estimate,
    make_model_spec,
    make_parallel_dims,
    parallelize_fake_model,
)


@pytest.fixture(autouse=True)
def _reset_distributed():
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def test_estimate_memory_returns_expected_categories():
    spec = make_model_spec("debugmodel", seq_len=128)
    model, fake_mode = build_fake_model(spec.model, dtype=torch.bfloat16)
    pd = make_parallel_dims(world_size=8, tp=2)
    parallelize_fake_model(model, parallel_dims=pd)

    snap = estimate_memory(model, fake_mode, pd, batch_size=2, seq_len=64)

    # At least one device has a positive Total.
    assert any(b.get("Total", 0) > 0 for b in snap.values())

    # Gather non-Total category names across devices.
    names: set[str] = set()
    for breakdown in snap.values():
        for key in breakdown:
            if key != "Total":
                names.add(key.value if hasattr(key, "value") else str(key))

    for required in ("Sharded Param", "Sharded Grad", "Activation", "OptState"):
        assert required in names, f"missing category {required!r} in {names}"


def _toy_snapshot():
    """A hand-built snapshot in the format that ``estimate_memory`` returns."""
    return {
        torch.device("cuda:0"): {
            "Sharded Param": 1_000_000,
            "Activation": 500_000,
            "Sharded Grad": 0,  # should be skipped in output
            "Total": 1_500_000,
        },
        torch.device("cpu"): {
            "OptState": 4_000_000,
            "Total": 4_000_000,
        },
    }


def test_format_skips_zero_rows_and_shows_total():
    out = format_memory_estimate(_toy_snapshot(), units="MiB")

    assert "Sharded Param" in out
    assert "Activation" in out
    assert "OptState" in out
    # Zero row dropped.
    assert "Sharded Grad" not in out
    # Total row present for each device.
    assert out.count("Total") == 2


def test_format_unit_validation():
    with pytest.raises(ValueError):
        format_memory_estimate({}, units="not_a_unit")
