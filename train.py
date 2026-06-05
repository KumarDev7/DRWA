#!/usr/bin/env python3
"""DRWA Training Script — full JIT scan loop, bf16, remat, metrics."""

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from drwa.config import DRWAConfig, TrainConfig
from drwa.run_config import RunConfig, load_config
from drwa.model import DRWAModel, forward_and_loss, compute_step_flops, count_params
from drwa.data import create_data_loader
from drwa.checkpoint import CheckpointManager, MetricsLogger


def create_optimizer(model, config: TrainConfig):
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.lr_transformer,
        warmup_steps=config.warmup_steps,
        decay_steps=config.total_steps,
        end_value=config.lr_transformer * config.lr_min_scale,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.adamw(learning_rate=schedule, weight_decay=0.01),
    )
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


def train(config: RunConfig, resume_from: str = None):
    print("=" * 72)
    print("  DRWA Training")
    print("=" * 72)

    devices = jax.devices()
    device = devices[0]
    n_devices = len(devices)
    model_config = config.model
    train_config = config.train

    compute_dtype = jnp.bfloat16 if model_config.compute_dtype == "bfloat16" else jnp.float32
    print(f"  Device: {device.device_kind} (x{n_devices})")
    print(f"  Compute dtype: {compute_dtype}")
    print(f"  Remat: {model_config.remat}")

    rngs = nnx.Rngs(train_config.seed)
    model = DRWAModel(model_config, rngs=rngs)

    n_params = count_params(model)
    step_flops = compute_step_flops(model_config, train_config.batch_size)
    tokens_per_step = train_config.batch_size * model_config.seq_len
    steps_per_window = train_config.steps_per_window

    peak_tflops_map = {
        "v5e": 197.0, "v5lite": 197.0, "v5 lite": 197.0, "v5p": 459.0,
        "a100": 312.0, "h100": 990.0, "cpu": 0.0,
    }
    device_kind_lower = str(device.device_kind).lower().replace("(", "").replace(")", "")
    peak_tflops = 0.0
    for key, val in peak_tflops_map.items():
        if key.lower() in device_kind_lower:
            peak_tflops = val * n_devices
            break

    print(f"  Parameters: {n_params:,}")
    print(f"  FLOPs/step: {step_flops/1e9:.2f}G")
    print(f"  Tokens/step: {tokens_per_step:,}")
    print(f"  Peak TFLOP/s: {peak_tflops:.0f}")
    print(f"  Window: {steps_per_window} steps/JIT compile")
    print(f"  d_model={model_config.d_model}, d_ffn={model_config.d_ffn}, "
          f"layers={model_config.n_layers_A}A+{model_config.n_layers_B}B, "
          f"heads={model_config.n_heads}, kv={model_config.n_kv_heads}")

    optimizer = create_optimizer(model, train_config)

    ckpt_manager = CheckpointManager(
        checkpoint_dir=config.checkpoint.dir,
        max_to_keep=config.checkpoint.keep,
    )
    metrics_logger = MetricsLogger(
        log_dir=config.checkpoint.dir,
        use_wandb=True,
        wandb_project=config.wandb.project,
        wandb_name=config.wandb.name,
    )

    start_step = 0
    if resume_from:
        try:
            metadata = ckpt_manager.load(model, optimizer, step=None)
            start_step = metadata.get("step", 0)
            print(f"  Resumed from step {start_step}")
        except ValueError:
            print("  No checkpoint found, starting fresh")

    loader = create_data_loader(
        {"model": model_config.__dict__, "data": config.data.__dict__},
        train_config,
    )

    total_steps = train_config.total_steps
    log_every = train_config.log_every

    # === Compiled training functions ===
    # Single step (for steps_per_window=1)
    @nnx.jit
    def train_step(model, optimizer, batch):
        def loss_fn(m):
            return forward_and_loss(m, batch, model_config, train_config, deterministic=False)
        (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads)
        return loss, metrics

    # Multi-step window (for steps_per_window > 1) — entire loop inside scan
    if steps_per_window > 1:
        @nnx.jit
        def train_window(model, optimizer, data):
            # data: [steps_per_window, batch_size, seq_len]
            # Entire training loop runs inside jax.lax.scan — no Python loop
            def step_fn(carry, xs):
                m, opt = carry
                batch = xs
                def loss_fn(mdl):
                    return forward_and_loss(mdl, batch, model_config, train_config, deterministic=False)
                (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(m)
                opt.update(m, grads)
                return (m, opt), loss
            init_carry = (model, optimizer)
            _, losses = jax.lax.scan(step_fn, init_carry, data)
            return jnp.mean(losses)

    print(f"\n  Compiling...")
    warmup_data = loader.get_window(steps_per_window if steps_per_window > 1 else 1)
    if steps_per_window > 1:
        warmup_input = jnp.array(warmup_data)
    else:
        warmup_input = jnp.array(warmup_data[0])

    t0 = time.time()
    if steps_per_window > 1:
        _ = train_window(model, optimizer, warmup_input)
    else:
        _ = train_step(model, optimizer, warmup_input)
    jax.block_until_ready(1)
    compile_time = time.time() - t0
    print(f"  Compiled in {compile_time:.1f}s")

    print(f"\n  Training {total_steps} steps ({total_steps // steps_per_window} windows of {steps_per_window})")
    print("-" * 78)
    print(f"  {'step':>7} {'loss':>8} {'ppl':>8} {'tks/s':>8} {'TFLOP/s':>8} {'MFU%':>6} {'elapsed':>8}")
    print("-" * 78)

    losses = []
    ss_steps = 0
    ss_time = 0.0
    min_win_time = float("inf")
    start_time = time.time()

    if steps_per_window > 1:
        n_windows = (total_steps - start_step) // steps_per_window
        for window_idx in range(n_windows):
            step = start_step + window_idx * steps_per_window

            data = jnp.array(loader.get_window(steps_per_window))

            win_t0 = time.time()
            loss_val = float(jax.block_until_ready(train_window(model, optimizer, data)))
            win_time = time.time() - win_t0

            sps = steps_per_window / win_time if win_time > 0 else 0
            tps = sps * tokens_per_step
            tflops = step_flops * sps / 1e12 if sps > 0 else 0
            mfu = (tflops / peak_tflops * 100) if peak_tflops > 0 else 0

            is_compile = win_time > 3.0 * min_win_time if min_win_time < float("inf") else (window_idx < 1)
            if not is_compile:
                ss_steps += steps_per_window
                ss_time += win_time
            min_win_time = min(min_win_time, win_time)

            for _ in range(steps_per_window):
                losses.append(loss_val)

            if window_idx % max(1, log_every // steps_per_window) == 0:
                elapsed = time.time() - start_time
                window = min(len(losses), steps_per_window)
                avg_loss = np.mean(losses[-window:])
                print(
                    f"  {step:7d} {avg_loss:8.4f} {float(np.exp(avg_loss)):8.2f} "
                    f"{tps:8.0f} {tflops:8.2f} {mfu:5.1f}% "
                    f"{time.strftime('%H:%M:%S', time.gmtime(elapsed))}"
                )

            metrics_logger.log({
                "loss": loss_val, "perplexity": float(np.exp(loss_val)),
                "tokens_per_sec": tps, "tflops": tflops, "mfu_pct": mfu,
                "step_time_sec": win_time / steps_per_window,
            }, step=step)

            if step > 0 and step % config.checkpoint.every == 0:
                ckpt_manager.save(model, optimizer, step, metrics={"loss": loss_val})

        final_step = start_step + n_windows * steps_per_window
    else:
        for step in range(start_step, total_steps):
            batch = jnp.array(loader.get_window(1)[0])

            step_t0 = time.time()
            loss, metrics = train_step(model, optimizer, batch)
            loss_val = float(jax.block_until_ready(loss))
            step_time = time.time() - step_t0
            losses.append(loss_val)

            sps = 1.0 / step_time if step_time > 0 else 0
            tps = sps * tokens_per_step
            tflops = step_flops * sps / 1e12 if sps > 0 else 0
            mfu = (tflops / peak_tflops * 100) if peak_tflops > 0 else 0

            is_compile = step_time > 3.0 * min_win_time if min_win_time < float("inf") else (step < 1)
            if not is_compile:
                ss_steps += 1
                ss_time += step_time
            min_win_time = min(min_win_time, step_time)

            if step % log_every == 0:
                elapsed = time.time() - start_time
                window = min(len(losses), log_every) if log_every > 0 else len(losses)
                avg_loss = np.mean(losses[-window:])
                print(
                    f"  {step:7d} {avg_loss:8.4f} {float(np.exp(avg_loss)):8.2f} "
                    f"{tps:8.0f} {tflops:8.2f} {mfu:5.1f}% "
                    f"{time.strftime('%H:%M:%S', time.gmtime(elapsed))}"
                )

            metrics_logger.log({
                "loss": loss_val, "perplexity": float(np.exp(loss_val)),
                "tokens_per_sec": tps, "tflops": tflops, "mfu_pct": mfu,
                "step_time_sec": step_time,
            }, step=step)

            if step > 0 and step % config.checkpoint.every == 0:
                ckpt_manager.save(model, optimizer, step, metrics={"loss": loss_val})

        final_step = total_steps

    ckpt_manager.save(model, optimizer, final_step, metrics={"final": True})
    metrics_logger.save()
    metrics_logger.close()

    initial_loss = np.mean(losses[:min(20, len(losses))])
    final_loss = np.mean(losses[-min(20, len(losses)):])
    total_time = time.time() - start_time
    ss_sps = ss_steps / ss_time if ss_time > 0 else 0
    ss_tps = ss_sps * tokens_per_step
    ss_tflops = step_flops * ss_sps / 1e12 if ss_sps > 0 else 0
    ss_mfu = (ss_tflops / peak_tflops * 100) if peak_tflops > 0 else 0

    print("\n" + "=" * 78)
    print(f"  Training complete!")
    print(f"  Initial loss: {initial_loss:.4f}  (ppl: {np.exp(initial_loss):.1f})")
    print(f"  Final loss:   {final_loss:.4f}  (ppl: {np.exp(final_loss):.1f})")
    print(f"  Total time:   {total_time:.0f}s")
    if ss_time > 0:
        print(f"  Steady-state: {ss_sps:.1f} steps/s | {ss_tps:.0f} tks/s | {ss_tflops:.1f} TFLOP/s | {ss_mfu:.1f}% MFU")
    print(f"  Checkpoints:  {config.checkpoint.dir}")


def main():
    parser = argparse.ArgumentParser(description="Train DRWA model")
    parser.add_argument("--config", type=str, required=True,
                        help="Config YAML path or preset: dense_small, dense_medium, dense_large, dense_colab")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, resume_from=args.resume)


if __name__ == "__main__":
    main()