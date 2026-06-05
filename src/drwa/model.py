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
    ):
        self.d_A = d_A
        self.d_B = d_B
        self.r = r

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
        h_res = h_A @ self.W_base.value.T + (h_A @ self.V.value.T) @ self.U.value.T + (self.b_base.value + self.b.value)[None, None, :]
        h_mid = self.norm(h_proj + self.gamma.value[0] * h_res)
        return h_mid, {}


class DRWAModel(nnx.Module):
    def __init__(self, config: DRWAConfig, rngs: nnx.Rngs):
        self.config = config
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
        )

        self.assembly = SimpleAssembly(
            d_A=config.d_A,
            d_B=config.d_B,
            r=config.r,
            gamma_init=config.gamma_init,
            dtype=compute_dtype,
            rngs=rngs,
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
        )

        self.lm_head = nnx.Linear(config.d_B, config.vocab_size, rngs=rngs)

    def __call__(self, input_ids: jnp.ndarray, deterministic: bool = True) -> Tuple[jnp.ndarray, dict]:
        B, T = input_ids.shape

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

    attn_flops_per_layer = B * T * (
        4 * d * d
        + 2 * T * head_dim * n_heads
    )

    ffn_flops_per_layer = B * T * (
        2 * d * d_ffn
    )

    assembly_flops = B * T * (2 * config.d_A * config.d_B * config.r)

    embed_flops = B * T * d

    lm_head_flops = B * T * d * V

    forward_flops = (
        n_layers * (attn_flops_per_layer + ffn_flops_per_layer)
        + assembly_flops
        + embed_flops
        + lm_head_flops
    )

    return int(3 * forward_flops)


def count_params(model: nnx.Module) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))