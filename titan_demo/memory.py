"""Estimate per-rank peak memory for a parallelized fake model.

Wraps ``torch.distributed._tools.fsdp2_mem_tracker.FSDPMemTracker`` so
users can run a fake training step (forward + backward + optimizer.step())
on a model produced by ``parallelize_fake_model`` and read off the
per-category byte counts (sharded/unsharded params, sharded/unsharded
grads, activations, optimizer states, all-gather/reduce-scatter buffers).

Everything happens under the same ``FakeTensorMode`` used to build the
model; collectives go through the fake backend that ``FSDPMemTracker``
loads at import time.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.fsdp2_mem_tracker import FSDPMemTracker

from torchtitan.distributed.full_dtensor import parallelize_inputs
from torchtitan.distributed.parallel_dims import ParallelDims


__all__ = ["estimate_memory", "format_memory_estimate", "print_memory_estimate"]


# How many bytes in each unit. Used by format_memory_estimate.
_UNITS = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}

# Category key emitted by ``MemTracker`` for the per-device row total.
_TOTAL_KEY = "Total"

# Column widths for format_memory_estimate.
_NAME_COL = 18
_VALUE_COL = 10


def _category_name(key: object) -> str:
    """Render a snapshot key (enum or plain string) as a readable label."""
    # ``MemTracker`` keys categories with ``_FSDPRefType`` enum members
    # whose values are the human-readable names ("Sharded Param", "OptState"
    # etc.). The "Total" row uses a plain string instead.
    return key.value if isinstance(key, Enum) else str(key)


def estimate_memory(
    model: nn.Module,
    fake_mode: FakeTensorMode,
    parallel_dims: ParallelDims,
    *,
    batch_size: int = 1,
    seq_len: int = 2048,
    optimizer_cls: type[torch.optim.Optimizer] = torch.optim.AdamW,
    optimizer_kwargs: dict[str, Any] | None = None,
) -> dict[torch.device, dict[str, int]]:
    """Run one fake training step and return the peak memory snapshot.

    The step is forward -> ``logits.sum().backward()`` -> ``optimizer.step()``.
    ``logits.sum()`` is a stand-in for the loss; we only care about the
    shapes that drive the FSDP all-gather / reduce-scatter buffers and the
    optimizer-state allocation.

    Args:
        model: Output of ``parallelize_fake_model``. Must be FSDP-wrapped
            (FSDPMemTracker asserts this).
        fake_mode: The ``FakeTensorMode`` from ``build_fake_model``. Every
            op in this function runs inside it so no real memory is
            allocated.
        parallel_dims: The same ``ParallelDims`` passed to
            ``parallelize_fake_model``. Used to wrap the fake tokens as a
            DTensor on the model's SPMD mesh (full_dtensor mode rejects
            plain tensors at the forward boundary).
        batch_size: Per-rank input batch size. The DP axes shard the
            global batch, so the global batch is
            ``batch_size * dp_replicate * dp_shard``.
        seq_len: Fake input sequence length. Must be
            ``<= model.rope.max_seq_len`` set via ``make_model_spec``.
        optimizer_cls: Optimizer class to instantiate over
            ``model.parameters()``. Defaults to AdamW. Choice affects the
            ``OptState`` category (Adam needs 2 fp32 states per param;
            SGD needs 0 or 1).
        optimizer_kwargs: Forwarded to ``optimizer_cls``. Default ``lr=1e-3``.

    Returns:
        ``{torch.device: {category: bytes}}``. Categories come from
        ``_FSDPRefType``: ``Sharded Param``, ``Unsharded Param``, ``Buffer``,
        ``Sharded Grad``, ``Unsharded Grad``, ``Activation``, ``Temp``,
        ``All Gather``, ``Reduce Scatter``, ``OptState``, ``Inputs``,
        plus the row total under ``"Total"``.
    """
    optimizer_kwargs = {"lr": 1e-3} if optimizer_kwargs is None else optimizer_kwargs

    with fake_mode:
        optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

        tokens = torch.empty((batch_size, seq_len), dtype=torch.long)
        # parallelize_inputs wants (inputs, labels) -- we reuse tokens for
        # labels and discard the wrapped labels.
        tokens_dt, _, _ = parallelize_inputs(parallel_dims, tokens, tokens, {})

        tracker = FSDPMemTracker(model, optimizer)
        tracker.track_inputs((tokens_dt,))
        with tracker:
            optimizer.zero_grad()
            logits = model(tokens_dt)
            loss = logits.sum()
            loss.backward()
            optimizer.step()

    return tracker.get_tracker_snapshot("peak")


def format_memory_estimate(
    snapshot: dict[torch.device, dict[str, int]],
    *,
    units: str = "MiB",
) -> str:
    """Format a memory snapshot as a per-device, per-category table.

    Args:
        snapshot: Output of ``estimate_memory``.
        units: ``"B"``, ``"KiB"``, ``"MiB"`` (default), or ``"GiB"``.
    """
    if units not in _UNITS:
        raise ValueError(f"Unknown units {units!r}; pick one of {list(_UNITS)}")
    factor = _UNITS[units]

    lines: list[str] = []
    for device, breakdown in snapshot.items():
        lines.append(f"=== {device} (peak) ===")
        # Sort categories by descending byte count, drop zero rows, put
        # "Total" last so readers can spot-check the sum.
        non_total = [
            (_category_name(k), v)
            for k, v in breakdown.items()
            if k != _TOTAL_KEY and v > 0
        ]
        non_total.sort(key=lambda kv: -kv[1])
        for name, bytes_val in non_total:
            lines.append(
                f"  {name:<{_NAME_COL}} : {bytes_val / factor:>{_VALUE_COL}.2f} {units}"
            )
        total = breakdown.get(_TOTAL_KEY, 0)
        lines.append(f"  {'-' * _NAME_COL}   {'-' * _VALUE_COL}")
        lines.append(
            f"  {'Total':<{_NAME_COL}} : {total / factor:>{_VALUE_COL}.2f} {units}"
        )
        lines.append("")
    return "\n".join(lines).rstrip()


def print_memory_estimate(
    snapshot: dict[torch.device, dict[str, int]],
    *,
    units: str = "MiB",
) -> None:
    """Print a memory snapshot in the format produced by ``format_memory_estimate``."""
    print(format_memory_estimate(snapshot, units=units))
