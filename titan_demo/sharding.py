"""Pretty-print TorchTitan model sharding configs for inspection.

Sharding declarations are spread across many sub-configs (every ``Linear``,
``RMSNorm``, attention block, FFN). A 70B model has hundreds of entries.
This helper prints only the structurally distinct parts:

  - Root buffer placement (``freqs_cis``)
  - ``tok_embeddings``, root ``norm``, ``lm_head``
  - One transformer layer (default layer 0)

All other layers share the layer-0 sharding plan.
"""

from __future__ import annotations

from typing import Any

from torch.distributed.tensor import Partial, Placement, Replicate, Shard

from torchtitan.models.common.attention import QKVLinear
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.protocols.sharding import ShardingConfig
from torchtitan.protocols.types import NamedPlacement


__all__ = ["format_sharding_config", "print_sharding_config"]


def _fmt_placement(p: Placement) -> str:
    if isinstance(p, Replicate):
        return "Replicate"
    if isinstance(p, Shard):
        return f"Shard({p.dim})"
    if isinstance(p, Partial):
        return "Partial"
    return repr(p)


def _fmt_named(named: NamedPlacement) -> str:
    return ", ".join(f"{axis.value}={_fmt_placement(p)}" for axis, p in named.items())


_FIELD_COL = 22


def _row(label: str, value: str) -> str:
    return f"  {label:<{_FIELD_COL}}: {value}"


def _fmt_sharding(sharding: ShardingConfig | None) -> list[str]:
    if sharding is None:
        return ["  <no sharding_config>"]

    lines: list[str] = []
    for name, np in sharding.state_shardings.items():
        lines.append(_row(f"state.{name}", _fmt_named(np)))
    if sharding.in_src_shardings:
        for name, np in sharding.in_src_shardings.items():
            lines.append(_row(f"in_src.{name}", _fmt_named(np)))
    if sharding.in_dst_shardings:
        for name, np in sharding.in_dst_shardings.items():
            lines.append(_row(f"in_dst.{name}", _fmt_named(np)))
    if sharding.out_src_shardings is not None:
        out_src = sharding.out_src_shardings
        if isinstance(out_src, tuple):
            for i, np in enumerate(out_src):
                lines.append(_row(f"out_src[{i}]", _fmt_named(np)))
        else:
            lines.append(_row("out_src", _fmt_named(out_src)))
    if sharding.out_dst_shardings is not None:
        lines.append(_row("out_dst", _fmt_named(sharding.out_dst_shardings)))
    if sharding.local_input_grad_placements:
        for name, np in sharding.local_input_grad_placements.items():
            lines.append(_row(f"in_grad.{name}", _fmt_named(np)))
    if sharding.local_output_grad_placements is not None:
        lines.append(
            _row("out_grad", _fmt_named(sharding.local_output_grad_placements))
        )
    if sharding.local_map is not None:
        for i, np in enumerate(sharding.local_map.in_grad_placements):
            value = _fmt_named(np) if np is not None else "<None>"
            lines.append(_row(f"local_map.in_grads[{i}]", value))

    if not lines:
        lines.append("  <empty sharding_config>")
    return lines


def _type_name(cfg: Any) -> str:
    """``Llama3Model.Config`` -> ``"Llama3Model"``."""
    return cfg.__class__.__qualname__.removesuffix(".Config")


def _format_entry(fqn: str, cfg: Any) -> list[str]:
    header = f"[{fqn}] {_type_name(cfg)}"
    sharding = getattr(cfg, "sharding_config", None)
    return [header, *_fmt_sharding(sharding)]


def _layer_entries(config: Any, layer_id: int) -> list[tuple[str, Any]]:
    layer = config.layers[layer_id]
    prefix = f"layers.{layer_id}"

    entries: list[tuple[str, Any]] = [
        (f"{prefix}.attention_norm", layer.attention_norm),
        (f"{prefix}.attention", layer.attention),
    ]

    qkv = layer.attention.qkv_linear
    if isinstance(qkv, QKVLinear.Config):
        entries += [
            (f"{prefix}.attention.qkv_linear.wq", qkv.wq),
            (f"{prefix}.attention.qkv_linear.wkv", qkv.wkv),
        ]
    else:
        # FusedQKVLinear.Config
        entries.append((f"{prefix}.attention.qkv_linear.wqkv", qkv.wqkv))

    entries += [
        (f"{prefix}.attention.wo", layer.attention.wo),
        (f"{prefix}.attention.inner_attention", layer.attention.inner_attention),
        (f"{prefix}.ffn_norm", layer.ffn_norm),
        (f"{prefix}.feed_forward", layer.feed_forward),
        (f"{prefix}.feed_forward.w1", layer.feed_forward.w1),
        (f"{prefix}.feed_forward.w2", layer.feed_forward.w2),
        (f"{prefix}.feed_forward.w3", layer.feed_forward.w3),
    ]
    return entries


def format_sharding_config(
    spec_or_config: ModelSpec | Any, *, layer_id: int = 0
) -> str:
    """Return the printable sharding-config snapshot as a string.

    Args:
        spec_or_config: Either a ``ModelSpec`` (the result of
            ``make_model_spec``) or the underlying ``Llama3Model.Config``.
        layer_id: Which transformer layer to print. Default 0. All other
            layers share the same sharding plan.
    """
    config = (
        spec_or_config.model
        if isinstance(spec_or_config, ModelSpec)
        else spec_or_config
    )

    sections: list[tuple[str, list[tuple[str, Any]]]] = [
        (
            "Root, Embedding, Norm, LM Head",
            [
                ("<root>", config),
                ("tok_embeddings", config.tok_embeddings),
                ("norm", config.norm),
                ("lm_head", config.lm_head),
            ],
        ),
        (
            f"Layer {layer_id} (all other layers share this plan)",
            _layer_entries(config, layer_id),
        ),
    ]

    out: list[str] = []
    for title, items in sections:
        out.append(f"=== {title} ===")
        for fqn, cfg in items:
            out.extend(_format_entry(fqn, cfg))
        out.append("")
    return "\n".join(out).rstrip()


def print_sharding_config(
    spec_or_config: ModelSpec | Any, *, layer_id: int = 0
) -> None:
    """Print the sharding-config snapshot for root, embedding/loss, and one layer."""
    print(format_sharding_config(spec_or_config, layer_id=layer_id))
