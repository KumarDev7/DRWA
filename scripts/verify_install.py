#!/usr/bin/env python3
"""
Script to verify DRWA installation and run a quick smoke test.

Usage:
    python scripts/verify_install.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from drwa.config import DRWAConfig, TrainConfig
from drwa.model import DRWAModel, forward_and_loss, compute_step_flops
from drwa.data import SyntheticDataLoader


def test_model_forward():
    """Test forward pass with N=1 dense mode."""
    print("Testing model forward pass (N=1 dense mode)...")
    
    config = DRWAConfig.dense_small()
    rngs = nnx.Rngs(42)
    
    model = DRWAModel(config, rngs=rngs)
    print(f"✓ Model created: {config.total_params / 1e6:.2f}M params")
    
    # Test forward pass
    batch_size = 2
    seq_len = config.seq_len
    input_ids = jax.random.randint(jax.random.PRNGKey(0), (batch_size, seq_len), 0, config.vocab_size)
    
    logits, metrics = model(input_ids, deterministic=True)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits: {logits.shape}")
    if metrics:
        metrics_str = ', '.join(f'{k}={jnp.mean(v):.3f}' for k, v in metrics.items())
        print(f"  Metrics: {metrics_str}")
    
    return True


def test_loss_backward():
    """Test loss computation and backward pass."""
    print("\nTesting loss + backward pass...")
    
    config = DRWAConfig.dense_small()
    rngs = nnx.Rngs(42)
    
    model = DRWAModel(config, rngs=rngs)
    
    batch_size = 4
    input_ids = jax.random.randint(jax.random.PRNGKey(1), (batch_size, config.seq_len), 0, config.vocab_size)
    
    loss, metrics = forward_and_loss(model, input_ids, config, TrainConfig(), deterministic=True)
    
    print(f"✓ Loss computed: {loss:.4f}")
    print(f"  Perplexity: {metrics['perplexity']:.2f}")
    
    # Test backward pass
    def loss_fn(m):
        return forward_and_loss(m, input_ids, config, TrainConfig(), deterministic=True)[0]
    
    grads = nnx.grad(loss_fn)(model)
    print(f"✓ Backward pass successful")
    
    return True


def test_data_loader():
    """Test synthetic data loader."""
    print("\nTesting data loader...")
    
    config = DRWAConfig.dense_small()
    
    loader = SyntheticDataLoader(
        seq_len=config.seq_len,
        batch_size=8,
        vocab_size=config.vocab_size,
        seed=42,
        pattern="random",
    )
    
    window = loader.get_window(steps=4)
    print(f"✓ Data loader working")
    print(f"  Window shape: {window.shape}")
    print(f"  Dtype: {window.dtype}")
    
    return True


def test_flops_calculation():
    """Test FLOPs calculation."""
    print("\nTesting FLOPs calculation...")
    
    config = DRWAConfig.dense_small()
    
    from drwa.config import TrainConfig
    train_config = TrainConfig(batch_size=32)
    
    flops = compute_step_flops(config, train_config.batch_size)
    print(f"✓ FLOPs per step: {flops / 1e9:.2f}G")
    
    return True


def main():
    print("=" * 80)
    print("DRWA Verification Script")
    print("=" * 80)
    
    tests = [
        ("Model Forward (N=1)", test_model_forward),
        ("Loss + Backward", test_loss_backward),
        ("Data Loader", test_data_loader),
        ("FLOPs Calc", test_flops_calculation),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
            print(f"✓ {name} PASSED")
        except Exception as e:
            failed += 1
            print(f"✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("=" * 80)
    print(f"Results: {passed}/{passed + failed} tests passed")
    
    if failed == 0:
        print("✓ All tests passed! DRWA framework is ready.")
        print("\nNext steps:")
        print("  1. Train dense model: python train.py --config configs/dense_small.yaml")
        print("  2. Expand pool: python scripts/expand_pool.py --checkpoint checkpoints/dense_small/model.ckpt --n-pool 1024")
        print("  3. Fine-tune DRWA: python train.py --config configs/drwa_1b.yaml --resume checkpoints/dense_small_N1024.ckpt")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())