"""titan_demo: scripts and APIs for demonstrating TorchTitan parallelism."""

# Apply PyTorch + TorchTitan monkey-patches before any code touches the
# patched call sites. See ``titan_demo/_patches.py`` for what and why.
from titan_demo._patches import apply_patches as _apply_patches

_apply_patches()

from titan_demo.fake_model import build_fake_model, make_model_spec  # noqa: E402
from titan_demo.memory import (  # noqa: E402
    estimate_memory,
    format_memory_estimate,
    print_memory_estimate,
)
from titan_demo.parallelize import (  # noqa: E402
    make_parallel_dims,
    parallelize_fake_model,
)
from titan_demo.sharding import (  # noqa: E402
    format_sharding_config,
    print_sharding_config,
)

__all__ = [
    "build_fake_model",
    "estimate_memory",
    "format_memory_estimate",
    "format_sharding_config",
    "make_model_spec",
    "make_parallel_dims",
    "parallelize_fake_model",
    "print_memory_estimate",
    "print_sharding_config",
]
