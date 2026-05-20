"""Monkey-patches for PyTorch and TorchTitan to support CP under fake mode.

Context Parallel (``cp > 1``) currently does not work end-to-end under
``FakeTensorMode`` with the upstream PyTorch + TorchTitan code. This
module applies the minimum set of in-process patches needed for
``parallelize_fake_model`` + ``estimate_memory`` to handle CP correctly.

Issues addressed:

1. ``torch.distributed.tensor.placement_types._StridedShard.local_shard_size_and_offset``
   builds a real-tensor ``torch.arange`` and reads the offsets via
   ``.tolist()``. Under FakeTensorMode the ``.tolist()`` triggers
   ``aten._local_scalar_dense`` which is data-dependent and refused.
   This path is hit by the view that ``nn.Linear`` does internally to
   flatten ``(B, S, dim)`` to ``(B*S, dim)`` when both ``B`` and ``S``
   are sharded on distinct mesh axes (DP_SHARD + CP) -- so every Linear
   in the model is affected.

2. ``FSDPMemTracker`` does not advertise ``supports_higher_order_operators``
   and does not handle HOOs in its ``__torch_dispatch__``. FlexAttention
   (required for CP) is a HOO and immediately errors.

3. ``FSDPMemTracker`` is not marked as an infra ``TorchDispatchMode``.
   FlexAttention internally uses ``torch.compile``, which bails when a
   non-infra mode is on the stack.

4. ``ModTracker`` registers global ``nn.Module`` forward hooks. When
   ``torch.compile`` runs the compiled artifact, it executes ephemeral
   ``torch.fx.GraphModule`` instances; their ``__call__`` fires those
   hooks and the surrounding AOT autograd queues a backward callback
   that resets ``ModTracker.parents`` to ``{"Global"}``. When the real
   enclosing module's post-hook fires, the pop raises
   ``"Module hierarchy tracking is wrong"``.

5. ``torchtitan.models.common.attention.FlexAttention.forward`` always
   calls the ``torch.compile``-wrapped flex_attention. Even with all
   the PyTorch-side patches in place, that compile path lowers to
   inductor / triton, which has no CPU backend and crashes during
   autotuning.

6. ``redistribute_cost`` invokes a graph-based Dijkstra
   (``generate_graph_based_transform_infos``) whenever a placement set
   contains ``_StridedShard``. For TP + CP combined, the strategy
   enumeration loop in ``expand_to_full_mesh_op_strategy`` calls it
   many times with complex 2-axis placements and the state space
   explodes (the call never returns within tens of minutes on a
   4-rank toy model). Under fake mode the *cost* is only used to rank
   candidate strategies; we short-circuit it to ``0.0`` for
   ``_StridedShard`` cases so strategy ranking degrades to
   first-viable-wins, but the execution-time redistribute path (a
   separate cached entry point) still runs the full Dijkstra and
   produces correct transforms. The earlier attempt that
   short-circuited ``_gen_transform_infos_non_cached`` directly also
   broke execution: empty transforms meant the actual redistribute
   was skipped, and downstream ops like ``lm_head``'s backward
   matmul saw shape mismatches.

The patches are read-only over the patched call sites' semantics: they
only redirect dispatch and skip phantom modules. For real-device runs
that import ``titan_demo``:

- Patches (1) and (5) always wrap the call, but the wrapper
  (``unset_fake_temporarily()`` / ``active_fake_mode()`` check) makes
  them no-ops outside ``FakeTensorMode``.
- Patch (6) is explicitly gated on ``active_fake_mode()``.
- Patches (2) and (3) install permanent class-level changes on
  ``FSDPMemTracker``; they only affect code that uses
  ``FSDPMemTracker`` and are additive there (HOO support, infra-mode
  flag).
- Patch (4) installs a permanent class-level change on ``ModTracker``
  but only fires for ``torch.fx.GraphModule`` instances, which are a
  compile-only artifact and never appear in plain ``ModTracker`` use.
"""

from __future__ import annotations

import torch


_APPLIED = False


def apply_patches() -> None:
    """Apply all CP-under-fake patches. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True

    _patch_strided_shard()
    _patch_fsdp_mem_tracker()
    _patch_mod_tracker()
    _patch_flex_attention()
    _patch_redistribute_cost()


def _patch_strided_shard() -> None:
    """Wrap ``_StridedShard.local_shard_size_and_offset`` in unset_fake."""
    from torch._guards import active_fake_mode
    from torch._subclasses.fake_tensor import unset_fake_temporarily
    from torch.distributed.tensor.placement_types import _StridedShard

    orig = _StridedShard.local_shard_size_and_offset

    def patched(self, *args, **kwargs):
        # Gate on active_fake_mode so real-device runs (every Linear when
        # CP is in play) skip the context-manager overhead entirely.
        if active_fake_mode():
            with unset_fake_temporarily():
                return orig(self, *args, **kwargs)
        return orig(self, *args, **kwargs)

    _StridedShard.local_shard_size_and_offset = patched


def _patch_fsdp_mem_tracker() -> None:
    """Teach ``FSDPMemTracker`` to accept HOOs and act as an infra mode."""
    from functools import partial

    from torch.distributed._tools.fsdp2_mem_tracker import (
        _FSDPModState,
        _FSDPRefType,
        FSDPMemTracker,
    )
    from torch.distributed.tensor import DTensor
    from torch.utils._pytree import tree_map_only

    FSDPMemTracker.supports_higher_order_operators = True
    FSDPMemTracker.is_infra_mode = classmethod(lambda _cls: True)

    orig_dispatch = FSDPMemTracker.__torch_dispatch__

    # args=... (Ellipsis) matches FSDPMemTracker's upstream signature:
    #   def __torch_dispatch__(self, func, types, args=..., kwargs=None)
    # It is the upstream default, not a placeholder.
    def patched_dispatch(self, func, types, args=..., kwargs=None):
        # Mirror upstream: let DTensor desugar first if it is on the stack.
        # FlexAttention currently unwraps before the HOO fires, so this is
        # not exercised today, but skipping the check would silently
        # diverge from the upstream contract for any future HOO that runs
        # while DTensor is still a subclass.
        if any(t == DTensor for t in types):
            return NotImplemented
        if isinstance(func, torch._ops.HigherOrderOperator):
            # Snapshot ModTracker state so AOT autograd queued during the
            # HOO cannot reset it under us. The GraphModule skip in
            # _patch_mod_tracker handles the main path; this is belt-and-
            # suspenders for any other compile-generated activity.
            saved_parents = self._mod_tracker.parents.copy()
            saved_active = dict(self._mod_tracker._active_module_cnt)
            try:
                res = func(*args, **(kwargs or {}))
            finally:
                self._mod_tracker.parents = saved_parents
                self._mod_tracker._active_module_cnt = saved_active

            # Mirror the three-branch reftype selection from the upstream
            # __torch_dispatch__ so a future HOO inside optimizer.step gets
            # categorized as OptState, not Activation.
            if self._in_opt:
                reftype = _FSDPRefType.OPT
            elif self._mod_tracker.is_bw and not self._in_ac:
                reftype = _FSDPRefType.TEMP
            else:
                reftype = _FSDPRefType.ACT
            # Use _track (the same call upstream __torch_dispatch__ makes)
            # so storage-size changes via _update_snap(SIZE, ...) are picked
            # up; _update_and_maybe_create_winfos only reacts to reftype
            # changes and would miss an in-place HOO storage resize.
            tree_map_only(torch.Tensor, partial(self._track, reftype), res)
            # Refresh the peak snapshot like the upstream non-HOO path
            # does; without this, a HOO that allocates the high-water
            # mark would not update the peak until the next regular op.
            peak_state = (
                _FSDPModState.PEAK_BW
                if self._mod_tracker.is_bw
                else _FSDPModState.PEAK_FW
            )
            self._update_peak_stats(peak_state)
            return res
        return orig_dispatch(self, func, types, args, kwargs)

    FSDPMemTracker.__torch_dispatch__ = patched_dispatch


def _patch_mod_tracker() -> None:
    """Skip ``torch.fx.GraphModule`` instances in ModTracker hooks."""
    import torch.fx
    from torch.distributed._tools.mod_tracker import ModTracker

    orig_pre = ModTracker._fw_pre_hook
    orig_post = ModTracker._fw_post_hook

    def should_skip(mod):
        return isinstance(mod, torch.fx.GraphModule)

    def patched_pre(self, mod, input):
        if should_skip(mod):
            return
        return orig_pre(self, mod, input)

    def patched_post(self, mod, input, output):
        if should_skip(mod):
            return
        return orig_post(self, mod, input, output)

    ModTracker._fw_pre_hook = patched_pre
    ModTracker._fw_post_hook = patched_post


def _patch_flex_attention() -> None:
    """Use eager flex_attention under fake mode (bypasses torch.compile)."""
    from torch._guards import active_fake_mode
    from torch.nn.attention.flex_attention import flex_attention
    from torchtitan.models.common.attention import FlexAttention

    orig_compiled = FlexAttention._compiled_flex_attn

    def dispatch(*args, **kwargs):
        if active_fake_mode():
            return flex_attention(*args, **kwargs)
        return orig_compiled(*args, **kwargs)

    FlexAttention._compiled_flex_attn = staticmethod(dispatch)


def _patch_redistribute_cost() -> None:
    """Short-circuit ``redistribute_cost`` for ``_StridedShard`` cases under
    fake mode so strategy enumeration doesn't run the slow Dijkstra."""
    from torch._guards import active_fake_mode
    from torch.distributed.tensor import _collective_utils
    from torch.distributed.tensor._ops import utils as _ops_utils
    from torch.distributed.tensor.placement_types import _StridedShard

    orig = _collective_utils.redistribute_cost

    def patched(current_spec, target_spec):
        # Check active_fake_mode() first so real-device callers skip the
        # placement scan entirely.
        if active_fake_mode() and any(
            isinstance(p, _StridedShard)
            for p in (*current_spec.placements, *target_spec.placements)
        ):
            # Cost is only used to rank candidate strategies in
            # expand_to_full_mesh_op_strategy. Returning 0 makes all
            # _StridedShard strategies tie (first-viable-wins). The
            # actual redistribute happens through a separate cached
            # entry point in DTensor's dispatch path, which still runs
            # the full Dijkstra and produces correct transforms.
            return 0.0
        return orig(current_spec, target_spec)

    # Patch both the canonical definition and the eagerly-bound copy on
    # the call-site module (_ops.utils does
    # ``from ... import redistribute_cost`` at module level).
    _collective_utils.redistribute_cost = patched
    _ops_utils.redistribute_cost = patched
