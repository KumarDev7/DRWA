#!/usr/bin/env python3
"""
Performance profiling tools for DRWA.

Measures FLOPs, TFLOPS, MFU, and per-component timing.
"""

import time
import sys
from pathlib import Path
from typing import Dict, Any

import jax
import jax.numpy as jnp
from flax import nnx

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from drwa.config import DRWAConfig, TrainConfig
from drwa.model import DRWAModel, forward_and_loss, compute_step_flops


def profile_forward_pass(
    model: DRWAModel,
    config: DRWAConfig,
    train_config: TrainConfig,
    num_iterations: int = 10,
    warmup_iterations: int = 2,
) -> Dict[str, Any]:
    """Profile forward pass timing.
    
    Args:
        model: DRWA model
        config: Model config
        train_config: Training config
        num_iterations: Number of timing iterations
        warmup_iterations: Warmup iterations (not timed)
    
    Returns:
        Dict with timing metrics
    """
    batch_size = train_config.batch_size
    seq_len = config.seq_len
    
    # Generate random input
    key = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(key, (batch_size, seq_len), 0, config.vocab_size)
    
    # Compile forward pass
    @nnx.jit
    def forward_step(model, input_ids):
        return forward_and_loss(model, input_ids, config, train_config, deterministic=True)
    
    # Warmup
    print(f"Warming up ({warmup_iterations} iterations)...")
    for _ in range(warmup_iterations):
        loss, metrics = forward_step(model, input_ids)
        jax.block_until_ready(loss)
    
    # Timed iterations
    print(f"Timing forward pass ({num_iterations} iterations)...")
    times = []
    for _ in range(num_iterations):
        start = time.time()
        loss, metrics = forward_step(model, input_ids)
        jax.block_until_ready(loss)
        end = time.time()
        times.append(end - start)
    
    # Compute stats
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    
    return {
        "forward_time_mean": mean_time,
        "forward_time_std": std_time,
        "forward_time_min": min(times),
        "forward_time_max": max(times),
    }


def profile_backward_pass(
    model: DRWAModel,
    config: DRWAConfig,
    train_config: TrainConfig,
    num_iterations: int = 10,
    warmup_iterations: int = 2,
) -> Dict[str, Any]:
    """Profile backward pass timing.
    
    Args:
        model: DRWA model
        config: Model config
        train_config: Training config
        num_iterations: Number of timing iterations
        warmup_iterations: Warmup iterations (not timed)
    
    Returns:
        Dict with timing metrics
    """
    batch_size = train_config.batch_size
    seq_len = config.seq_len
    
    # Generate random input
    key = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(key, (batch_size, seq_len), 0, config.vocab_size)
    
    # Compile backward pass
    @nnx.jit
    def backward_step(model, input_ids):
        def loss_fn(m):
            return forward_and_loss(m, input_ids, config, train_config, deterministic=True)
        (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        return loss, grads
    
    # Warmup
    print(f"Warming up ({warmup_iterations} iterations)...")
    for _ in range(warmup_iterations):
        loss, grads = backward_step(model, input_ids)
        jax.block_until_ready(loss)
    
    # Timed iterations
    print(f"Timing backward pass ({num_iterations} iterations)...")
    times = []
    for _ in range(num_iterations):
        start = time.time()
        loss, grads = backward_step(model, input_ids)
        jax.block_until_ready(loss)
        end = time.time()
        times.append(end - start)
    
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    
    return {
        "backward_time_mean": mean_time,
        "backward_time_std": std_time,
        "backward_time_min": min(times),
        "backward_time_max": max(times),
    }


def compute_throughput(
    config: DRWAConfig,
    train_config: TrainConfig,
    forward_time: float,
    backward_time: float,
) -> Dict[str, Any]:
    """Compute throughput metrics.
    
    Args:
        config: Model config
        train_config: Training config
        forward_time: Mean forward pass time (seconds)
        backward_time: Mean backward pass time (seconds)
    
    Returns:
        Dict with throughput metrics
    """
    step_flops = compute_step_flops(config, train_config.batch_size)
    total_time = forward_time + backward_time
    
    steps_per_sec = 1.0 / total_time
    tflops_per_sec = step_flops * steps_per_sec / 1e12
    tokens_per_sec = train_config.batch_size * config.seq_len * steps_per_sec
    
    # Assume TPU v5e peak = 197 TFLOPS (bf16)
    # This is approximate; adjust for your hardware
    tpu_v5e_peak_tflops = 197.0
    mfu = (tflops_per_sec / tpu_v5e_peak_tflops) * 100
    
    return {
        "step_flops": step_flops,
        "forward_time_sec": forward_time,
        "backward_time_sec": backward_time,
        "total_time_sec": total_time,
        "steps_per_sec": steps_per_sec,
        "tflops_per_sec": tflops_per_sec,
        "tokens_per_sec": int(tokens_per_sec),
        "mfu_percent": mfu,
        "model_params": config.total_params,
    }


def profile_model(
    config: DRWAConfig,
    train_config: TrainConfig,
    seed: int = 42,
) -> Dict[str, Any]:
    """Full model profile.
    
    Args:
        config: Model config
        train_config: Training config
        seed: Random seed
    
    Returns:
        Dict with all profiling metrics
    """
    print("=" * 80)
    print("DRWA Model Profiling")
    print("=" * 80)
    
    # Initialize JAX
    devices = jax.devices()
    print(f"Devices: {len(devices)} x {devices[0].device_kind}")
    
    # Create model
    print("Initializing model...")
    rngs = nnx.Rngs(seed)
    model = DRWAModel(config, rngs=rngs)
    
    # Count parameters
    model_state = nnx.state(model, nnx.Param)
    param_count = sum(
        x.size for x in jax.tree_util.tree_leaves(model_state)
    )
    print(f"✓ Model parameters: {param_count / 1e6:.2f}M")
    
    # Profile forward pass
    print("\n--- Forward Pass ---")
    forward_metrics = profile_forward_pass(model, config, train_config)
    for k, v in forward_metrics.items():
        if "time" in k:
            print(f"  {k}: {v * 1000:.2f} ms")
        else:
            print(f"  {k}: {v}")
    
    # Profile backward pass
    print("\n--- Backward Pass ---")
    backward_metrics = profile_backward_pass(model, config, train_config)
    for k, v in backward_metrics.items():
        if "time" in k:
            print(f"  {k}: {v * 1000:.2f} ms")
        else:
            print(f"  {k}: {v}")
    
    # Compute throughput
    print("\n--- Throughput ---")
    throughput_metrics = compute_throughput(
        config,
        train_config,
        forward_metrics["forward_time_mean"],
        backward_metrics["backward_time_mean"],
    )
    print(f"  Steps/sec: {throughput_metrics['steps_per_sec']:.2f}")
    print(f"  TFLOPS: {throughput_metrics['tflops_per_sec']:.2f}")
    print(f"  Tokens/sec: {throughput_metrics['tokens_per_sec']:,}")
    print(f"  MFU: {throughput_metrics['mfu_percent']:.1f}%")
    
    # Combine all metrics
    all_metrics = {
        **forward_metrics,
        **backward_metrics,
        **throughput_metrics,
    }
    
    print("=" * 80)
    
    return all_metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Profile DRWA model performance")
    parser.add_argument("--config", type=str, default="dense_small", help="Config preset or YAML path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Load config
    from drwa.run_config import load_config
    config = load_config(args.config)
    
    # Run profiling
    metrics = profile_model(config.model, config.train, seed=args.seed)
    
    # Print summary
    print("\n" + "=" * 80)
    print("PROFILING SUMMARY")
    print("=" * 80)
    print(f"Model params: {metrics['model_params'] / 1e6:.2f}M")
    print(f"Forward time: {metrics['forward_time_mean'] * 1000:.2f} ms")
    print(f"Backward time: {metrics['backward_time_mean'] * 1000:.2f} ms")
    print(f"Total time: {metrics['total_time_sec'] * 1000:.2f} ms")
    print(f"Throughput: {metrics['steps_per_sec']:.2f} steps/s")
    print(f"Performance: {metrics['tflops_per_sec']:.2f} TFLOP/s")
    print(f"Efficiency: {metrics['mfu_percent']:.1f}% MFU")
    print(f"Tokens/sec: {metrics['tokens_per_sec']:,}")
    print("=" * 80)


if __name__ == "__main__":
    main()