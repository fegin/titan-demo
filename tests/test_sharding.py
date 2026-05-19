"""Tests for ``titan_demo.format_sharding_config`` / ``print_sharding_config``."""

from __future__ import annotations

import torch

from titan_demo import format_sharding_config, make_model_spec


def test_format_covers_root_embedding_and_one_layer():
    spec = make_model_spec("debugmodel", seq_len=128)
    out = format_sharding_config(spec)

    # Top-level entries
    for fqn in ("<root>", "tok_embeddings", "norm", "lm_head"):
        assert f"[{fqn}]" in out, f"missing {fqn}"

    # Layer 0 entries
    for suffix in (
        "attention_norm",
        "attention.qkv_linear.wq",
        "attention.qkv_linear.wkv",
        "attention.wo",
        "attention.inner_attention",
        "ffn_norm",
        "feed_forward.w1",
        "feed_forward.w2",
        "feed_forward.w3",
    ):
        assert f"[layers.0.{suffix}]" in out, f"missing layers.0.{suffix}"

    # No other layers
    assert "[layers.1." not in out


def test_format_axes_visible_in_output():
    spec = make_model_spec("debugmodel", seq_len=128)
    out = format_sharding_config(spec)

    # Every dense entry should carry all four full_dtensor dense axes.
    for axis in ("dp_replicate", "dp_shard", "cp", "tp"):
        assert f"{axis}=" in out

    # The colwise/rowwise/SP placements should be visible.
    assert "Shard(0)" in out  # colwise weight
    assert "Shard(1)" in out  # rowwise weight / SP activation
    assert "Replicate" in out


def test_layer_id_selects_a_different_layer():
    spec = make_model_spec("debugmodel", seq_len=128)
    out = format_sharding_config(spec, layer_id=3)
    assert "[layers.3.attention.wo]" in out
    assert "[layers.0.attention.wo]" not in out


def test_local_map_in_grads_printed_per_input():
    spec = make_model_spec("debugmodel", seq_len=128)
    out = format_sharding_config(spec)
    # ScaledDotProductAttention sets local_map.in_grad_placements for (q, k, v).
    for i in range(3):
        assert f"local_map.in_grads[{i}]" in out
