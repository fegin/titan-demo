"""Build and run TorchTitan models under FakeTensorMode.

A fake-tensor build avoids real parameter allocation, so the demo runs
on a CPU-only machine (e.g., Colab) and large flavors like 70B or 405B
fit in seconds. Only shape and dtype metadata is tracked; values are
never materialized.

Parameters land on the cpu device (as FakeTensors). We deliberately do
**not** combine FakeTensorMode with ``torch.device("meta")`` even though
both achieve "no real allocation": FSDP's ``_validate_no_meta_params``
refuses meta-device params, which would block ``parallelize_fake_model``
and ``estimate_memory``.

The build is split into two steps so users can inspect or modify the
model config (sharding declarations, parallelism options, attention
backend, etc.) before any parameters are allocated:

    spec = make_model_spec("debugmodel", seq_len=128)
    # ... modify spec.model.layers[i].attention.sharding_config etc.
    model, fake_mode = build_fake_model(spec.model)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from torchtitan.models.llama3 import model_registry
from torchtitan.models.llama3.sharding import set_llama3_sharding_config
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.model_spec import ModelSpec


__all__ = ["make_model_spec", "build_fake_model"]


@contextmanager
def _default_dtype(dtype: torch.dtype) -> Iterator[None]:
    """Temporarily swap ``torch.get_default_dtype()`` for the body.

    TorchTitan modules read the default dtype at construction time
    (e.g., ``nn.Linear`` uses it to pick the parameter dtype), so we
    set it around ``config.build()``.
    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def _fakeify_module_tensors(module: nn.Module, fake_mode: FakeTensorMode) -> None:
    """Convert all params and buffers to FakeTensors, sharing aliases.

    Models built under ``torch.device("meta")`` already have meta-device
    parameters, but those are not FakeTensors and do not participate in
    FakeTensorMode's shape/dtype propagation during forward. We swap
    them in-place so that running forward inside ``fake_mode`` produces
    FakeTensor activations end-to-end.

    Shared parameters (e.g., weight-tied ``tok_embeddings`` and
    ``lm_head``) are detected by Python ``id()`` and only converted
    once so the alias survives.
    """
    memo: dict[int, torch.Tensor] = {}

    def as_fake(tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(tensor, FakeTensor):
            return tensor
        key = id(tensor)
        if key not in memo:
            memo[key] = fake_mode.from_tensor(tensor)
        return memo[key]

    for child in module.modules():
        for name, param in list(child._parameters.items()):
            if param is None:
                continue
            fake_param = as_fake(param)
            if fake_param is not param:
                child._parameters[name] = nn.Parameter(
                    fake_param, requires_grad=param.requires_grad
                )
        for name, buffer in list(child._buffers.items()):
            if buffer is None:
                continue
            child._buffers[name] = as_fake(buffer)


def make_model_spec(
    flavor: str = "debugmodel",
    *,
    seq_len: int = 2048,
    attn_backend: str = "sdpa",
    loss_parallel: bool = True,
    enable_sp: bool = True,
) -> ModelSpec:
    """Return the Llama3 ``ModelSpec`` for ``flavor`` without instantiating the model.

    The returned spec carries:
      - ``spec.model``: the ``Llama3Model.Config`` -- modify this to set
        sharding declarations, swap inner attention, change vocab size, etc.
      - ``spec.name`` / ``spec.flavor``: identifiers for logging.
      - ``spec.parallelize_fn``: the legacy parallelize callable.

    Default sharding configs are installed via ``set_llama3_sharding_config``
    so the spec is ready to feed to ``parallelize_fake_model``. Users can
    still override individual declarations by editing
    ``spec.model.layers[i].attention.sharding_config`` etc. before
    ``build_fake_model``.

    Args:
        flavor: One of ``"debugmodel"``, ``"1B"``, ``"3B"``, ``"8B"``,
            ``"70B"``, ``"405B"``. See ``torchtitan.models.llama3.llama3_configs``.
        seq_len: Sets ``spec.model.rope.max_seq_len`` so the RoPE cache is
            large enough for the planned forward pass.
        attn_backend: ``"sdpa"``, ``"flex"``, or ``"varlen"``.
        loss_parallel: If True (default), shard the output projection along
            the vocab dimension. Same toggle as
            ``ParallelismConfig.disable_loss_parallel`` (inverted).
        enable_sp: If True (default), enable SequenceParallel: norms are
            replicated and activations between TP regions are Shard(1) on
            the sequence dim. Same toggle as
            ``ParallelismConfig.enable_sequence_parallel``.
    """
    spec = model_registry(flavor, attn_backend=attn_backend)
    spec.model.rope.max_seq_len = max(seq_len, 1)
    set_llama3_sharding_config(
        spec.model, loss_parallel=loss_parallel, enable_sp=enable_sp
    )
    return spec


def build_fake_model(
    model_config: BaseModel.Config,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[nn.Module, FakeTensorMode]:
    """Instantiate ``model_config`` with FakeTensor params and buffers.

    Call ``make_model_spec`` first, modify ``spec.model`` (e.g. install a
    custom sharding config), then pass ``spec.model`` here.

    Args:
        model_config: A ``BaseModel.Config`` (typically ``spec.model``
            from ``make_model_spec``).
        dtype: Default dtype for parameters and buffers.

    Returns:
        ``(model, fake_mode)``. Pass ``fake_mode`` to downstream callers
        (e.g. ``estimate_memory``) so the forward pass uses the same mode.
    """
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

    with fake_mode, _default_dtype(dtype):
        model = model_config.build()

    with fake_mode:
        _fakeify_module_tensors(model, fake_mode)

    return model, fake_mode
