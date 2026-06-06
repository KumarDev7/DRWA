"""
DRWA Model Architecture.

Dense mode (N=1) for stable training, expandable to DRWA (N>1) after training.
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
from typing import Tuple, Optional

from .config import DRWAConfig, TrainConfig
from .parts import PartA, PartB


class SimpleAssembly(nnx.Module):
    def __init__(
        self,
        d_A: int,
        d_B: int,
        r: int,
        gamma_init: float,
        dtype=jnp.float32,
        rngs: nnx.Rngs = None,
        model_axis: str = None,
    ):
        self.d_A = d_A
        self.d_B = d_B
        self.r = r
        self.model_axis = model_axis

        self.W_base = nnx.Param(
            jax.random.normal(rngs.params(), (d_B, d_A), dtype=dtype) * 0.02
        )
        self.b_base = nnx.Param(jnp.zeros(d_B, dtype=dtype))
        self.gamma = nnx.Param(jnp.array([gamma_init], dtype=dtype))

        self.U = nnx.Param(
            jax.random.normal(rngs.params(), (d_B, r), dtype=dtype) * (d_A ** -0.5)
        )
        self.V = nnx.Param(
            jax.random.normal(rngs.params(), (r, d_A), dtype=dtype) * (d_A ** -0.5)
        )
        self.b = nnx.Param(jnp.zeros(d_B, dtype=dtype))

        self.norm = nnx.LayerNorm(d_B, dtype=dtype, rngs=rngs)
        if d_A != d_B:
            self.proj = nnx.Linear(d_A, d_B, use_bias=False, dtype=dtype, rngs=rngs)

    def __call__(self, h_A):
        h_proj = self.proj(h_A) if self.d_A != self.d_B else h_A
        # Assembly weights are replicated across the model axis (see sharding.py),
        # so h_res is computed identically on every device — no collective needed.
        h_res = h_A @ self.W_base.value.T + (h_A @ self.V.value.T) @ self.U.value.T + (self.b_base.value + self.b.value)[None, None, :]
        h_mid = self.norm(h_proj + self.gamma.value[0] * h_res)
        return h_mid, {}


class DRWAModel(nnx.Module):
    def __init__(self, config: DRWAConfig, rngs: nnx.Rngs, model_axis: str = None):
        self.config = config
        self.model_axis = model_axis
        compute_dtype = jnp.bfloat16 if config.compute_dtype == "bfloat16" else jnp.float32

        self.embed = nnx.Embed(config.vocab_size, config.d_model, dtype=compute_dtype, rngs=rngs)

        self.part_a = PartA(
            d_model=config.d_A,
            n_layers=config.n_layers_A,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            d_ffn=config.d_ffn,
            max_seq_len=config.seq_len,
            use_rope=config.use_rope,
            dtype=compute_dtype,
            remat=config.remat,
            rngs=rngs,
            model_axis=model_axis,
        )

        self.assembly = SimpleAssembly(
            d_A=config.d_A,
            d_B=config.d_B,
            r=config.r,
            gamma_init=config.gamma_init,
            dtype=compute_dtype,
            rngs=rngs,
            model_axis=model_axis,
        )

        self.part_b = PartB(
            d_model=config.d_B,
            n_layers=config.n_layers_B,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            d_ffn=config.d_ffn,
            max_seq_len=config.seq_len,
            use_rope=config.use_rope,
            dtype=compute_dtype,
            remat=config.remat,
            rngs=rngs,
            model_axis=model_axis,
        )

        self.lm_head = nnx.Linear(config.d_B, config.vocab_size, dtype=compute_dtype, rngs=rngs)

    def __call__(self, input_ids: jnp.ndarray, deterministic: bool = True) -> Tuple[jnp.ndarray, dict]:
        B, T = input_ids.shape

        # Embedding is replicated across the model axis (see sharding.py), so the
        # output is already the full, replicated d_model on every device.
        h = self.embed(input_ids)

        h_A = self.part_a(h, deterministic=deterministic)
        h_mid, assembly_info = self.assembly(h_A)
        h_B = self.part_b(h_mid, deterministic=deterministic)

        logits = self.lm_head(h_B)

        metrics = {
            "h_A_norm": jnp.linalg.norm(h_A, axis=-1).mean(),
            "h_mid_norm": jnp.linalg.norm(h_mid, axis=-1).mean(),
            "h_B_norm": jnp.linalg.norm(h_B, axis=-1).mean(),
            **assembly_info,
        }

        return logits, metrics


@nnx.jit
def _gen_forward_jit(model: "DRWAModel", ctx: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled single forward pass for generation.

    Defined at module level so the compiled XLA kernel is cached and reused
    across every token step and every call to generate().  The function must
    always receive ctx with the same shape ([1, seq_len]) — see generate().

    Under GSPMD (outside shard_map) psum calls are skipped via try/except;
    JAX's transparent sharding handles the necessary collectives automatically.
    """
    logits, _ = model(ctx, deterministic=True)
    return logits[:, -1, :].astype(jnp.float32)  # [1, vocab] — last token only


def generate(
    model: "DRWAModel",
    prompt_ids: jnp.ndarray,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    rng: jnp.ndarray = None,
) -> jnp.ndarray:
    """Autoregressive generation with optional top-p (nucleus) sampling."""
    seq_len = model.config.seq_len
    ids = prompt_ids  # 1-D

    for _ in range(max_new_tokens):
        # Always pad-left to exactly seq_len so _gen_forward_jit's compiled
        # XLA kernel is reused every iteration (shape never changes).
        ctx = ids[-seq_len:]
        if ctx.shape[0] < seq_len:
            ctx = jnp.concatenate([jnp.zeros(seq_len - ctx.shape[0], dtype=jnp.int32), ctx])
        next_logits = _gen_forward_jit(model, ctx[None, :])[0]  # [vocab]

        if rng is None:
            next_token = jnp.argmax(next_logits)
        else:
            next_logits = next_logits / max(temperature, 1e-6)
            if top_p < 1.0:
                sorted_idx = jnp.argsort(-next_logits)
                sorted_logits = next_logits[sorted_idx]
                probs = jax.nn.softmax(sorted_logits)
                cum_probs = jnp.cumsum(probs)
                mask = cum_probs - probs > top_p
                sorted_logits = jnp.where(mask, -1e9, sorted_logits)
                next_logits = next_logits.at[sorted_idx].set(sorted_logits)
            rng, sample_rng = jax.random.split(rng)
            next_token = jax.random.categorical(sample_rng, next_logits)

        ids = jnp.concatenate([ids, next_token[None]])

    return ids


_gen_compiled = False  # module-level flag to print compilation warning once


def _generate_step(model, config, tokenizer, prompt, max_new_tokens, temperature, top_p, rng, step):
    """Run generation and print samples; called from the training loop."""
    import numpy as np
    global _gen_compiled

    print(f"\n  --- generation @ step {step} ---")
    if not _gen_compiled:
        print(f"  (compiling generation kernel for seq_len={config.seq_len}...)")
        _gen_compiled = True
    for i, p in enumerate(prompt):
        enc = tokenizer.encode(p)
        prompt_ids = jnp.array(enc, dtype=jnp.int32)
        key = jax.random.fold_in(rng, i) if rng is not None else None
        out_ids = generate(model, prompt_ids, max_new_tokens, temperature, top_p, key)
        out_ids_np = np.array(out_ids).tolist()
        text = tokenizer.decode(out_ids_np)
        label = f"  [{i}] prompt: {p!r}"
        print(label)
        print(f"       output: {text!r}")
    print(f"  --- end generation ---\n")


def forward_and_loss(
    model: DRWAModel,
    input_ids: jnp.ndarray,
    config: DRWAConfig,
    train_config: TrainConfig,
    deterministic: bool = False,
) -> Tuple[jnp.ndarray, dict]:
    B, T = input_ids.shape

    logits, metrics = model(input_ids, deterministic=deterministic)

    logits_shifted = logits[:, :-1, :]
    targets_shifted = input_ids[:, 1:]

    logits_flat = logits_shifted.reshape(-1, config.vocab_size)
    targets_flat = targets_shifted.reshape(-1)

    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits_flat.astype(jnp.float32), targets_flat
    ).mean()

    metrics["loss"] = loss
    metrics["perplexity"] = jnp.exp(loss)

    return loss, metrics


def compute_step_flops(config: DRWAConfig, batch_size: int) -> int:
    B = batch_size
    T = config.seq_len
    d = config.d_model
    d_ffn = config.d_ffn
    V = config.vocab_size
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = d // n_heads
    n_layers = config.n_layers_A + config.n_layers_B

    d_kv = n_kv_heads * head_dim
    d_q = n_heads * head_dim
    attn_proj_flops = 2 * B * T * (d * d_q + d * d_kv + d * d_kv + d * d_q)
    attn_score_flops = 2 * B * T * T * n_heads * head_dim
    attn_output_flops = attn_score_flops
    attn_flops_per_layer = attn_proj_flops + attn_score_flops + attn_output_flops

    ffn_flops_per_layer = 2 * B * T * (d * d_ffn + d_ffn * d)

    assembly_flops = 2 * B * T * (config.d_A * config.d_B + config.d_A * config.r + config.r * config.d_B)

    embed_flops = B * T * d

    lm_head_flops = 2 * B * T * d * V

    forward_flops = (
        n_layers * (attn_flops_per_layer + ffn_flops_per_layer)
        + assembly_flops
        + embed_flops
        + lm_head_flops
    )

    return int(3 * forward_flops)


def count_params(model: nnx.Module) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))