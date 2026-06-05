#!/usr/bin/env python3
"""Colab TPU setup and training launcher for DRWA.

Run this in a Google Colab cell with TPU v5e-1 runtime:
  !pip install -q jax[tpu] flax optax transformers
  !git clone <your-repo> && cd drwa
  !python scripts/colab_setup.py --config dense_colab
"""

import os
import sys
import subprocess
import argparse


def check_tpu():
    import jax
    devices = jax.devices()
    print(f"Devices: {len(devices)} x {devices[0].device_kind}")
    if "tpu" not in str(devices[0].device_kind).lower() and "v5" not in str(devices[0].device_kind).lower():
        print("WARNING: No TPU detected. Training will run on CPU/GPU — much slower.")
    else:
        hbm_gb = devices[0].memory_capacity / 1e9 if hasattr(devices[0], 'memory_capacity') else 16
        print(f"HBM: {hbm_gb:.0f} GB")


def install_deps():
    packages = [
        "jax[tpu]",
        "flax",
        "optax",
        "transformers",
        "datasets",
        "wandb",
        "pyyaml",
    ]
    print("Installing latest versions...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U"] + packages)


def train(config_name: str, resume: str = None):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from drwa.config import DRWAConfig
    from drwa.model import DRWAModel, compute_step_flops, count_params
    from drwa.run_config import RunConfig

    presets = {
        "dense_small": DRWAConfig.dense_small,
        "dense_medium": DRWAConfig.dense_medium,
        "dense_large": DRWAConfig.dense_large,
        "dense_colab": DRWAConfig.dense_colab,
    }

    if config_name in presets:
        config = RunConfig(model=presets[config_name]())
    elif os.path.exists(config_name):
        config = RunConfig.load(config_name)
    else:
        raise ValueError(f"Unknown config: {config_name}")

    from train import train as train_fn
    train_fn(config, resume_from=resume)


def main():
    parser = argparse.ArgumentParser(description="DRWA Colab Setup & Training")
    parser.add_argument("--config", type=str, default="dense_colab",
                        help="Config: dense_small, dense_medium, dense_large, dense_colab, or YAML path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--skip-install", action="store_true", help="Skip dependency installation")
    parser.add_argument("--check-only", action="store_true", help="Only check TPU, don't train")
    args = parser.parse_args()

    if not args.skip_install:
        install_deps()

    check_tpu()

    if args.check_only:
        return

    train(args.config, args.resume)


if __name__ == "__main__":
    main()