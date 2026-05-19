"""titan_demo: scripts and APIs for demonstrating TorchTitan parallelism."""

from titan_demo.fake_model import build_fake_model, make_model_spec
from titan_demo.memory import (
    estimate_memory,
    format_memory_estimate,
    print_memory_estimate,
)
from titan_demo.parallelize import make_parallel_dims, parallelize_fake_model
from titan_demo.sharding import format_sharding_config, print_sharding_config

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
