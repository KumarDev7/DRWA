#!/usr/bin/env python3
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import optax

from drwa.config import DRWAConfig, TrainConfig
from drwa.model import DRWAModel, forward_and_loss


def generate_text(model, tokenizer, prompt, max_new_tokens=40, temperature=0.8, top_k=40, rng_seed=0):
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    generated = list(input_ids)

    for i in range(max_new_tokens):
        ctx = generated[-128:]
        logits, _ = model(jnp.array([ctx], dtype=jnp.int32), deterministic=True)
        next_logits = np.array(logits[0, -1])

        if temperature > 0:
            next_logits = next_logits / temperature

        if top_k > 0:
            top_k_indices = np.argsort(next_logits)[-top_k:]
            mask = np.full_like(next_logits, -np.inf)
            mask[top_k_indices] = next_logits[top_k_indices]
            next_logits = mask

        probs = jax.nn.softmax(jnp.array(next_logits))
        key = jax.random.PRNGKey(rng_seed + i)
        next_token = int(jax.random.categorical(key, jnp.array(next_logits)))
        generated.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated)


def main():
    print("=" * 70)
    print("DRWA CPU Training Test — TinyStories")
    print("=" * 70)

    device = jax.devices()[0]
    print(f"Device: {device.device_kind}")

    config = DRWAConfig(
        vocab_size=50257,
        seq_len=128,
        d_model=64,
        d_ffn=256,
        d_A=64,
        d_B=64,
        N=1,
        k_max=1,
        r=16,
        n_layers_A=2,
        n_layers_B=2,
        n_heads=4,
        n_kv_heads=4,
        gamma_init=0.05,
    )
    print(f"d_model={config.d_model}, d_ffn={config.d_ffn}, "
          f"layers={config.n_layers_A}+{config.n_layers_B}")

    model = DRWAModel(config, rngs=nnx.Rngs(42))
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model)))
    print(f"Parameters: {total_params:,}")

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1e-3, weight_decay=0.01),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    print("\nLoading TinyStories dataset...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    from drwa.data import TinyStoriesDataLoader
    loader = TinyStoriesDataLoader(
        seq_len=config.seq_len,
        batch_size=4,
        vocab_size=config.vocab_size,
        tokenizer_name="gpt2",
        seed=42,
    )
    print("Dataset loaded!")

    @nnx.jit
    def train_step(model, optimizer, batch):
        def loss_fn(m):
            loss, metrics = forward_and_loss(m, batch, config, TrainConfig(), deterministic=False)
            return loss, metrics
        (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads)
        return loss, metrics

    print("\nCompiling first step (this takes a minute on CPU)...")
    warmup_batch = jnp.array(loader.get_window(1)[0])
    t0 = time.time()
    _ = train_step(model, optimizer, warmup_batch)
    jax.block_until_ready(_)
    print(f"Compilation took {time.time()-t0:.1f}s")

    total_steps = 200
    log_every = 5
    generate_every = 50
    losses = []

    print("\n" + "=" * 70)
    print("TRAINING START")
    print("=" * 70)
    start_time = time.time()

    for step in range(total_steps):
        batch = jnp.array(loader.get_window(1)[0])
        loss, metrics = train_step(model, optimizer, batch)
        loss_val = float(loss)
        losses.append(loss_val)

        if step % log_every == 0:
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            window = min(len(losses), log_every)
            avg_loss = np.mean(losses[-window:])
            print(f"step {step:4d}/{total_steps} | loss={loss_val:.4f} | avg={avg_loss:.4f} | "
                  f"steps/s={steps_per_sec:.1f} | elapsed={elapsed:.0f}s")

        if step > 0 and step % generate_every == 0:
            prompt = "Once upon a time"
            try:
                text = generate_text(model, tokenizer, prompt, max_new_tokens=25, temperature=0.8, top_k=40, rng_seed=step)
                print(f"  [{prompt}] → {text[:120]}...")
            except Exception as e:
                print(f"  Generation error: {e}")

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    initial_loss = np.mean(losses[:10])
    final_loss = np.mean(losses[-10:])
    print(f"\nInitial loss (avg first 10): {initial_loss:.4f}")
    print(f"Final loss   (avg last 10):  {final_loss:.4f}")
    print(f"Loss reduction:              {initial_loss - final_loss:.4f}")

    if final_loss < initial_loss:
        print("\n✓ Loss IS reducing — model is learning!")
    else:
        print("\n✗ Loss NOT reducing — something is wrong")

    print("\n--- Generation samples ---")
    for prompt in ["Once upon a time", "A little girl", "The cat"]:
        try:
            text = generate_text(model, tokenizer, prompt, max_new_tokens=40, temperature=0.7, top_k=40, rng_seed=999)
            print(f"  [{prompt}] → {text[:150]}")
        except Exception as e:
            print(f"  [{prompt}] → (error: {e})")

    print("\nDone!")


if __name__ == "__main__":
    main()