"""
DRWA (Disk-Resident Reasoning Weight-Assembly) Training Framework

A JAX/Flax NNX-based training framework for the DRWA architecture,
optimized for TPU with multi-step windowing, Pallas kernels, and
model-parallel pool sharding.
"""

__version__ = "0.1.0"

from .model import DRWAModel, forward_and_loss, compute_step_flops, count_params
from .config import DRWAConfig, TrainConfig
from .run_config import RunConfig, load_config, save_config
from .sharding import create_mesh, get_param_sharding, shard_model, get_data_shardings, shard_data

__all__ = [
    "DRWAModel",
    "DRWAConfig",
    "TrainConfig",
    "RunConfig",
    "forward_and_loss",
    "compute_step_flops",
    "count_params",
    "load_config",
    "save_config",
    "create_mesh",
    "get_param_sharding",
    "shard_model",
    "get_data_shardings",
    "shard_data",
]