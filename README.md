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

# 6. Run a forward pass with fake tokens.
logits = run_forward(model, fake_mode, batch_size=2, seq_len=64)
print("logits:", tuple(logits.shape), logits.dtype)
```

Parameters and activations are FakeTensors on the meta device. No real
memory is allocated, so 70B and 405B builds finish in seconds on a
single CPU.

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
