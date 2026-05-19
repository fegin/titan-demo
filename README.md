# titan-demo

Scripts and APIs for demonstrating TorchTitan parallelism on CPU-only
environments (e.g., Google Colab) via PyTorch's FakeTensorMode.

## Install (from a notebook)

```python
!pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
!pip install git+https://github.com/pytorch/torchtitan.git
!pip install git+https://github.com/<your-user>/titan-demo.git
```

## Quick start

```python
import torch
from titan_demo import (
    make_model_spec,
    build_fake_model,
    make_parallel_dims,
    parallelize_fake_model,
    estimate_memory,
    print_memory_estimate,
    run_forward,
)

# 1. Get the spec (with default sharding declarations installed).
spec = make_model_spec(
    "debugmodel",         # or "1B", "3B", "8B", "70B", "405B"
    seq_len=128,
)

# 2. (Optional) inspect the default sharding plan, then tweak spec.model
#    if you want to override anything.
from titan_demo import print_sharding_config
print_sharding_config(spec)              # root + embedding/loss + layer 0
# print_sharding_config(spec, layer_id=1)  # any other layer if needed

# 3. Build under FakeTensorMode (no real memory).
model, fake_mode = build_fake_model(spec.model, dtype=torch.bfloat16)
print(f"{spec.flavor}: {sum(p.numel() for p in model.parameters()):,} params")

# 4. Pick parallelism degrees. world_size required; dp_shard=-1 fills the
#    leftover. full_dtensor is always True.
parallel_dims = make_parallel_dims(world_size=8, tp=2)  # -> dp_shard=4

# 5. Apply TP/CP sharding + fully_shard wrapping. Auto-inits a fake
#    process group of size world_size on first call.
parallelize_fake_model(model, spec=spec, parallel_dims=parallel_dims)

# 6. Estimate per-rank peak memory (params, grads, opt-state, acts,
#    all-gather / reduce-scatter buffers). Internally runs one full
#    fake training step (forward + backward + AdamW.step) under
#    torch.distributed._tools.fsdp2_mem_tracker.FSDPMemTracker.
snap = estimate_memory(model, fake_mode, parallel_dims, batch_size=2, seq_len=64)
print_memory_estimate(snap, units="MiB")
```

Parameters and activations are FakeTensors. No real memory is
allocated, so 70B and 405B builds finish in seconds on a single CPU.

## What the memory estimate represents

The estimate models **eager training** memory: TorchTitan's standard config compiles only the FlexAttention kernel; the rest of the model (MLP, RMSNorm, etc.) runs eager, and our estimator measures the activations those eager ops allocate.

If you run real training with full-model ``torch.compile`` enabled, fused MLP / norm kernels can drop some of the intermediates we count, so the actual peak may be **lower** than this estimate. The demo does not model that.

The ``OptState`` category lands on cpu in the raw tracker snapshot (an artifact of fake AdamW). The notebook helper rewrites it to a deterministic ``local_params * 12`` bytes (AdamW master + exp_avg + exp_avg_sq, fp32) and attributes it to the GPU device.

## Notebook

`notebooks/parallelism_explorer.ipynb` walks through three scenarios (8B on 8 GPUs, 70B on 128 GPUs, 405B on 1024 GPUs) with interactive widgets for `tp`, `dp_replicate`, `batch_size`, `seq_len`. Pick a config, click "Run Interact", and see whether it FITS, is NEAR OOM (>=95% of per-GPU budget), or OOMs. Install the notebook extra (`pip install -e .[notebook]`) to get `ipywidgets`. Context Parallel (`cp > 1`) is not yet supported under fake mode -- see the notebook's "Limitations" section.

## Layout

```
titan_demo/        Python package: public API
tests/             pytest smoke tests
pyproject.toml     pip-installable metadata
```

## Tests

```bash
pip install -e .[test]
pytest
```

## Lint and format

Mirrors TorchTitan's setup (subset): ``ufmt`` (black + usort) and
``flake8`` with ``flake8-bugbear`` + ``pep8-naming``.

```bash
pip install -e .[lint]
pre-commit install            # auto-run on commit
pre-commit run --all-files    # one-shot
# or directly:
ufmt format titan_demo tests
flake8 titan_demo tests --config=.flake8
```
