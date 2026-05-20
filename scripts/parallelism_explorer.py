"""Estimate per-rank memory for a TorchTitan parallelism config.

CLI version of ``notebooks/parallelism_explorer.ipynb``. Defaults to
70B + FSDP + TP + CP on 128 GPUs x 80 GiB. Override any axis with the
matching flag.

Example invocations:

    # Default: 70B, world=128, tp=8, cp=2, dp_shard=8 (= 128 / 8 / 2).
    python scripts/parallelism_explorer.py

    # Long context, more CP.
    python scripts/parallelism_explorer.py --cp 4 --seq-len 16384

    # Pure FSDP for comparison.
    python scripts/parallelism_explorer.py --tp 1 --cp 1

    # 8B on 8 GPUs.
    python scripts/parallelism_explorer.py --flavor 8B --world-size 8 \\
        --gpu-mem-gib 40 --tp 2 --cp 2 --seq-len 4096
"""

from __future__ import annotations

import argparse
import sys

import torch
from torch.distributed.tensor import DTensor

from titan_demo import (
    build_fake_model,
    estimate_memory,
    format_memory_estimate,
    make_model_spec,
    make_parallel_dims,
    parallelize_fake_model,
)


# AdamW keeps three fp32 values per parameter: master weight,
# first moment (exp_avg), second moment (exp_avg_sq).
_ADAMW_BYTES_PER_PARAM = 12

_FLAVORS = ("debugmodel", "1B", "3B", "8B", "70B", "405B")
_UNITS = ("B", "KiB", "MiB", "GiB")


def _local_param_count(model: torch.nn.Module) -> int:
    """Per-rank parameter count, summing local storage of each DTensor."""
    total = 0
    for p in model.parameters():
        total += p.to_local().numel() if isinstance(p, DTensor) else p.numel()
    return total


def _correct_optstate(
    snap: dict[torch.device, dict[object, int]],
    model: torch.nn.Module,
) -> dict[torch.device, dict[object, int]]:
    """Replace the tracker's OptState with a deterministic value.

    FSDPMemTracker under FakeTensorMode reports OptState on the cpu
    device and at a size that does not match a real run: AdamW state
    creation does not always preserve DTensor sharding under fake mode.
    We replace it with ``local_params * 12`` (AdamW master + exp_avg +
    exp_avg_sq, all fp32) and attribute it to the rank's compute device.
    Under FakeTensorMode that device is reported as cpu (it would be cuda
    in real training).
    """
    corrected = _local_param_count(model) * _ADAMW_BYTES_PER_PARAM

    compute_dev = next((d for d in snap if d.type == "cuda"), None)
    if compute_dev is None:
        compute_dev = next(iter(snap))

    for dev in list(snap):
        breakdown = snap[dev]
        for key in list(breakdown):
            name = key.value if hasattr(key, "value") else str(key)
            if name == "OptState":
                old = breakdown.pop(key)
                breakdown["Total"] = breakdown.get("Total", 0) - old
        if dev != compute_dev and breakdown.get("Total", 0) == 0:
            del snap[dev]

    snap[compute_dev]["OptState"] = corrected
    snap[compute_dev]["Total"] = snap[compute_dev].get("Total", 0) + corrected
    return snap


def _per_rank_gib(snap: dict[torch.device, dict[object, int]]) -> float:
    """Sum totals across all devices to get the per-rank peak in GiB."""
    return sum(d.get("Total", 0) for d in snap.values()) / (1024**3)


def _verdict(used_gib: float, budget_gib: float) -> str:
    pct = used_gib / budget_gib * 100
    if used_gib > budget_gib:
        return f"OOM: need {used_gib:.2f} GiB, have {budget_gib} GiB ({pct:.0f}%)"
    if pct >= 95:
        return f"NEAR OOM: {used_gib:.2f} / {budget_gib} GiB ({pct:.1f}%)"
    return f"FITS: {used_gib:.2f} / {budget_gib} GiB ({pct:.1f}%)"


def _validate(args: argparse.Namespace) -> str | None:
    """Return an error message string, or None if the config is valid.

    Mirrors the checks ``ParallelDims`` and ``estimate_memory`` would
    fail on, but with friendlier messages aimed at CLI users.
    """
    product = args.tp * args.cp * args.dp_replicate
    if product > args.world_size:
        return (
            f"tp * cp * dp_replicate = {product} > "
            f"world_size = {args.world_size}"
        )
    if args.world_size % product != 0:
        return (
            f"world_size={args.world_size} not divisible by "
            f"tp * cp * dp_replicate = {product} "
            f"(no integer dp_shard fits)"
        )
    # seq_len must be divisible by tp * 2 * cp (SP needs tp, CP needs 2*cp).
    divisor = args.tp * 2 * args.cp
    if args.seq_len % divisor != 0:
        return (
            f"seq_len={args.seq_len} not divisible by tp * 2 * cp = "
            f"{divisor}"
        )
    return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--flavor", default="70B", choices=_FLAVORS)
    p.add_argument("--world-size", type=int, default=128)
    p.add_argument(
        "--gpu-mem-gib", type=float, default=80,
        help="Per-GPU memory budget for the FITS / NEAR OOM / OOM verdict.",
    )
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--cp", type=int, default=2)
    p.add_argument("--dp-replicate", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--seq-len", type=int, default=8192,
        help="Global sequence length; per-rank shard is seq_len // cp. "
             "Must be divisible by tp * 2 * cp.",
    )
    p.add_argument("--units", default="GiB", choices=_UNITS)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    err = _validate(args)
    if err:
        print(f"error: {err}", file=sys.stderr)
        sys.exit(2)

    # CP requires FlexAttention; SDPA is fine (and slightly faster to
    # build) when cp == 1.
    attn_backend = "flex" if args.cp > 1 else "sdpa"
    spec = make_model_spec(
        args.flavor, seq_len=args.seq_len, attn_backend=attn_backend
    )
    model, fake_mode = build_fake_model(spec.model, dtype=torch.bfloat16)
    pd = make_parallel_dims(
        world_size=args.world_size,
        tp=args.tp,
        cp=args.cp,
        dp_replicate=args.dp_replicate,
    )
    parallelize_fake_model(model, parallel_dims=pd)

    snap = estimate_memory(
        model, fake_mode, pd,
        batch_size=args.batch_size, seq_len=args.seq_len,
    )
    _correct_optstate(snap, model)

    used = _per_rank_gib(snap)
    global_batch = args.batch_size * pd.dp_replicate * pd.dp_shard

    print(
        f"Llama3 {args.flavor} on {args.world_size} GPUs x "
        f"{args.gpu_mem_gib} GiB"
    )
    print(
        f"  tp={args.tp}, cp={args.cp}, dp_replicate={args.dp_replicate}, "
        f"dp_shard={pd.dp_shard}"
    )
    print(
        f"  per-rank batch={args.batch_size}, seq_len={args.seq_len} "
        f"(local {args.seq_len // args.cp})"
    )
    print(f"  global batch={global_batch}")
    print()
    print(f"Verdict: {_verdict(used, args.gpu_mem_gib)}")
    print()
    print(format_memory_estimate(snap, units=args.units))


if __name__ == "__main__":
    main()
