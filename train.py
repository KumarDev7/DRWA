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
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

shard_map = jax.shard_map

sys.path.insert(0, str(Path(__file__).parent / "src"))

from drwa.config import DRWAConfig, TrainConfig
from drwa.run_config import RunConfig, load_config
from drwa.model import DRWAModel, forward_and_loss, compute_step_flops, count_params
from drwa.sharding import create_mesh, shard_model, get_data_shardings
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
    n_devices = len(devices)
    device = devices[0]

    model_config = config.model
    train_config = config.train

    # Create 1D or 2D device mesh based on sharding config.
    # 1D (n_model=1): data-parallel only, all weights replicated.
    # 2D (n_model>1): data parallel on 'data' axis, model parallel on 'model' axis.
    mesh, mesh_info = create_mesh(config.sharding, devices)
    n_data = mesh_info["n_data"]
    n_model = mesh_info["n_model"]
    is_2d = mesh_info["is_2d"]

    # Validate batch divides across data-parallel axis
    if is_2d:
        if train_config.batch_size % n_data != 0:
            print(f"  WARNING: batch_size={train_config.batch_size} not divisible by n_data={n_data}; "
                  f"replicating data across data axis.")
        n_sharded = n_data if train_config.batch_size % n_data == 0 else 1
    else:
        batch_divides = train_config.batch_size % n_devices == 0
        n_sharded = n_devices if batch_divides else 1

    data_sharding, window_sharding = get_data_shardings(mesh)

    compute_dtype = jnp.bfloat16 if model_config.compute_dtype == "bfloat16" else jnp.float32
    sharding_type = f"2D ({n_data} data x {n_model} model)" if is_2d else f"1D (data x {n_sharded})"
    print(f"  Device: {device.device_kind} (x{n_devices})")
    print(f"  Compute dtype: {compute_dtype}")
    print(f"  Remat: {model_config.remat}")
    print(f"  Sharding: {sharding_type}")

    rngs = nnx.Rngs(train_config.seed)
    model_axis_name = 'model' if is_2d else None
    model = DRWAModel(model_config, rngs=rngs, model_axis=model_axis_name)

    # Apply model sharding for 2D meshes (no-op for 1D)
    if is_2d:
        model = shard_model(model, mesh)

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
            # step_flops is the *global* per-step compute; under both data- and
            # model-parallelism every device contributes FLOPs, so the aggregate
            # peak is per-chip-peak times the total device count.
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
        process_index=jax.process_index(),
        process_count=jax.process_count(),
    )

    total_steps = train_config.total_steps
    log_every = train_config.log_every

    with jax.set_mesh(mesh):

        graphdef, state0 = nnx.split((model, optimizer))

        def _state_specs(state):
            def _leaf_spec(x):
                if hasattr(x, 'sharding') and isinstance(x.sharding, NamedSharding):
                    return x.sharding.spec
                return P()
            orig_treedef = jax.tree_util.tree_structure(state)
            leaves = jax.tree_util.tree_leaves(state)
            specs = [_leaf_spec(leaf) for leaf in leaves]
            return jax.tree_util.tree_unflatten(orig_treedef, specs)

        state_specs = _state_specs(state0)
        pmean_fn = lambda x: jax.lax.pmean(x, axis_name='data')

        if steps_per_window <= 1:
            def _train_step_core(state, batch):
                m, opt = nnx.merge(graphdef, state)
                (loss, metrics), grads = nnx.value_and_grad(
                    lambda mdl: forward_and_loss(mdl, batch, model_config, train_config, deterministic=False),
                    has_aux=True,
                )(m)
                grads = jax.tree_util.tree_map(pmean_fn, grads)
                loss = pmean_fn(loss)
                metrics = jax.tree_util.tree_map(pmean_fn, metrics)
                opt.update(m, grads)
                _, new_state = nnx.split((m, opt))
                return new_state, loss, metrics

            train_step_core = jax.jit(shard_map(
                _train_step_core, mesh=mesh,
                in_specs=(state_specs, P('data', None)),
                out_specs=(state_specs, P(), P()),
                check_vma=False,
            ), donate_argnums=(0,))

            def train_step(model, optimizer, batch):
                _, state = nnx.split((model, optimizer))
                new_state, loss, metrics = train_step_core(state, batch)
                nnx.update((model, optimizer), new_state)
                return loss, metrics

        if steps_per_window > 1:
            def _train_window_core(state, data):
                def step_fn(carry_state, batch):
                    m, opt = nnx.merge(graphdef, carry_state)
                    (loss, metrics), grads = nnx.value_and_grad(
                        lambda mdl: forward_and_loss(mdl, batch, model_config, train_config, deterministic=False),
                        has_aux=True,
                    )(m)
                    grads = jax.tree_util.tree_map(pmean_fn, grads)
                    loss = pmean_fn(loss)
                    metrics = jax.tree_util.tree_map(pmean_fn, metrics)
                    opt.update(m, grads)
                    _, new_state = nnx.split((m, opt))
                    return new_state, (loss, metrics)
                final_state, (losses, metrics) = jax.lax.scan(step_fn, state, data)
                return final_state, jnp.mean(losses), metrics

            train_window_core = jax.jit(shard_map(
                _train_window_core, mesh=mesh,
                in_specs=(state_specs, P(None, 'data', None)),
                out_specs=(state_specs, P(), P()),
                check_vma=False,
            ), donate_argnums=(0,))

            def train_window(model, optimizer, data):
                _, state = nnx.split((model, optimizer))
                new_state, loss, metrics = train_window_core(state, data)
                nnx.update((model, optimizer), new_state)
                return loss, metrics

        print(f"\n  Compiling...")
        warmup_data = loader.get_window(steps_per_window if steps_per_window > 1 else 1)
        if steps_per_window > 1:
            warmup_input = jax.device_put(warmup_data, window_sharding)
        else:
            warmup_input = jax.device_put(warmup_data[0], data_sharding)

        t0 = time.time()
        if steps_per_window > 1:
            _ = train_window(model, optimizer, warmup_input)
        else:
            _ = train_step(model, optimizer, warmup_input)
        jax.block_until_ready(_)
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
            data_np = loader.get_window(steps_per_window)
            data = jax.device_put(data_np, window_sharding)

            for window_idx in range(n_windows):
                step = start_step + window_idx * steps_per_window

                win_t0 = time.time()
                loss, window_metrics = train_window(model, optimizer, data)
                loss_val = float(jax.block_until_ready(loss))
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

                if window_idx < n_windows - 1:
                    data_np = loader.get_window(steps_per_window)
                    data = jax.device_put(data_np, window_sharding)

            final_step = start_step + n_windows * steps_per_window
        else:
            batch_np = loader.get_window(1)
            for step in range(start_step, total_steps):
                batch = jax.device_put(batch_np[0], data_sharding)

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

                if step < total_steps - 1:
                    batch_np = loader.get_window(1)

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
                        help="Config YAML path or preset: dense_small, dense_medium, dense_large, dense_xl, dense_colab")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, resume_from=args.resume)


if __name__ == "__main__":
    main()