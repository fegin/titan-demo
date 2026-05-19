"""Apply TorchTitan parallelism to a fake-tensor model.

Workflow:

    spec = make_model_spec("debugmodel", seq_len=128)
    model, fake_mode = build_fake_model(spec.model)
    parallel_dims = make_parallel_dims(world_size=8, tp=2)
    parallelize_fake_model(model, spec=spec, parallel_dims=parallel_dims)

``parallelize_fake_model`` mutates ``model`` in place:
  1. (One-shot) initializes a ``"fake"`` process group of size
     ``parallel_dims.world_size`` so device-mesh construction works in a
     single CPU process.
  2. Calls ``parallel_dims.build_mesh()``.
  3. Calls ``model.parallelize(parallel_dims)`` for TP/CP sharding declared
     on the model config.
  4. Wraps modules with ``fully_shard`` for FSDP/HSDP.

Re-parallelizing the same model with different ``parallel_dims`` is not
supported -- rebuild via ``build_fake_model`` first.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

from torchtitan.distributed.full_dtensor import resolve_fsdp_mesh, validate_config
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.llama3.parallelize import apply_fsdp
from torchtitan.protocols.model_spec import ModelSpec


__all__ = ["make_parallel_dims", "setup_fake_distributed", "parallelize_fake_model"]


def make_parallel_dims(
    *,
    world_size: int,
    dp_replicate: int = 1,
    dp_shard: int = -1,
    cp: int = 1,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
) -> ParallelDims:
    """Construct a ``ParallelDims`` for the titan-demo (full_dtensor mode).

    ``dp_shard=-1`` fills leftover ranks after
    ``dp_replicate * cp * tp * pp``. So
    ``make_parallel_dims(world_size=8, tp=2)`` yields ``dp_shard=4``.

    Constraint: ``dp_replicate * dp_shard * cp * tp * pp == world_size``.
    """
    return ParallelDims(
        dp_replicate=dp_replicate,
        dp_shard=dp_shard,
        cp=cp,
        tp=tp,
        pp=pp,
        ep=ep,
        world_size=world_size,
        full_dtensor=True,
    )


def setup_fake_distributed(world_size: int) -> None:
    """Initialize a ``"fake"`` process group of size ``world_size``.

    Idempotent: returns immediately if ``torch.distributed`` is already
    initialized. Always uses ``rank=0`` because everything runs in one
    Python process; the fake backend does not actually transfer data.
    """
    if dist.is_initialized():
        return
    dist.init_process_group("fake", rank=0, world_size=world_size)


def parallelize_fake_model(
    model: nn.Module,
    *,
    spec: ModelSpec,
    parallel_dims: ParallelDims,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
    cpu_offload: bool = False,
) -> nn.Module:
    """Apply TP/CP sharding and FSDP wrapping to a fake-tensor model.

    Args:
        model: Model returned by ``build_fake_model``.
        spec: Model spec returned by ``make_model_spec`` (used to dispatch
            the per-model FSDP wrapper).
        parallel_dims: Constructed via ``make_parallel_dims``.
        param_dtype: ``MixedPrecisionPolicy.param_dtype`` for FSDP.
        reduce_dtype: ``MixedPrecisionPolicy.reduce_dtype`` for FSDP.
        cpu_offload: If True, FSDP offloads params/grads/opt-state to CPU.

    Returns:
        The same ``model`` (mutated in place), for convenience.
    """
    setup_fake_distributed(parallel_dims.world_size)
    parallel_dims.build_mesh()

    validate_config(parallel_dims, model)
    model.parallelize(parallel_dims)

    dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
    apply_fsdp(
        model,
        dp_mesh,
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=cpu_offload,
        dp_mesh_dims=dp_mesh_dims,
    )

    # Suppress unused-arg warning -- spec is reserved for future per-model
    # FSDP dispatch (Qwen3, DeepSeek-V3, etc.). Today we hard-code llama3.
    del spec

    return model
