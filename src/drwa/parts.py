"""Transformer building blocks with optional gradient checkpointing."""

import jax
import jax.numpy as jnp
from flax import nnx


def precompute_rope_freqs(dim: int, max_seq_len: int, base: float = 10000.0):
    freqs = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    positions = jnp.arange(max_seq_len, dtype=jnp.float32)
    angles = positions[:, None] * freqs[None, :]
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(x, cos, sin):
    B, T, n_heads, head_dim = x.shape
    cos = cos[:T, :][None, :, None, :]
    sin = sin[:T, :][None, :, None, :]
    half_dim = head_dim // 2
    x_rotated = jnp.concatenate([-x[..., half_dim:], x[..., :half_dim]], axis=-1)
    return x * cos + x_rotated * sin


class CausalSelfAttention(nnx.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, max_seq_len, use_rope=True, dtype=jnp.float32, rngs=None):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope
        self.dtype = dtype
        self.wq = nnx.Linear(d_model, n_heads * self.head_dim, dtype=dtype, rngs=rngs)
        self.wk = nnx.Linear(d_model, n_kv_heads * self.head_dim, dtype=dtype, rngs=rngs)
        self.wv = nnx.Linear(d_model, n_kv_heads * self.head_dim, dtype=dtype, rngs=rngs)
        self.wo = nnx.Linear(n_heads * self.head_dim, d_model, dtype=dtype, rngs=rngs)
        if use_rope:
            cos, sin = precompute_rope_freqs(self.head_dim, max_seq_len)
            cos_full = jnp.concatenate([cos, cos], axis=-1).astype(dtype)
            sin_full = jnp.concatenate([sin, sin], axis=-1).astype(dtype)
            self.cos = nnx.Variable(cos_full)
            self.sin = nnx.Variable(sin_full)

    def __call__(self, x, mask=None, deterministic=True):
        B, T, d = x.shape
        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).reshape(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).reshape(B, T, self.n_kv_heads, self.head_dim)
        if self.use_rope:
            q = apply_rope(q, self.cos.value, self.sin.value)
            k = apply_rope(k, self.cos.value, self.sin.value)
        out = jax.nn.dot_product_attention(
            q, k, v,
            is_causal=True,
            mask=mask,
        )
        out = out.reshape(B, T, self.n_heads * self.head_dim)
        return self.wo(out)


class FFN(nnx.Module):
    def __init__(self, d_model, d_ffn, dtype=jnp.float32, rngs=None):
        self.w1 = nnx.Linear(d_model, d_ffn, dtype=dtype, rngs=rngs)
        self.w2 = nnx.Linear(d_ffn, d_model, dtype=dtype, rngs=rngs)
        self.dtype = dtype

    def __call__(self, x):
        return self.w2(jax.nn.gelu(self.w1(x)))


class TransformerBlock(nnx.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, d_ffn, max_seq_len, use_rope=True, dtype=jnp.float32, remat=False, rngs=None):
        self.norm1 = nnx.LayerNorm(d_model, dtype=dtype, rngs=rngs)
        self.norm2 = nnx.LayerNorm(d_model, dtype=dtype, rngs=rngs)
        self.attn = CausalSelfAttention(d_model, n_heads, n_kv_heads, max_seq_len, use_rope, dtype, rngs)
        self.ffn = FFN(d_model, d_ffn, dtype, rngs)
        self._remat = remat

    def __call__(self, x, mask=None, deterministic=True):
        if self._remat:
            @nnx.remat
            def remat_block(block, h_in, m_in):
                h_out = h_in + block.attn(block.norm1(h_in), m_in, deterministic)
                h_out = h_out + block.ffn(block.norm2(h_out))
                return h_out
            x = remat_block(self, x, mask)
        else:
            x = x + self.attn(self.norm1(x), mask, deterministic)
            x = x + self.ffn(self.norm2(x))
        return x


class PartA(nnx.Module):
    def __init__(self, d_model, n_layers, n_heads, n_kv_heads, d_ffn, max_seq_len, use_rope=True, dtype=jnp.float32, remat=False, rngs=None):
        self.layers = nnx.List([
            TransformerBlock(d_model, n_heads, n_kv_heads, d_ffn, max_seq_len, use_rope, dtype, remat, rngs)
            for _ in range(n_layers)
        ])
        self.norm = nnx.LayerNorm(d_model, dtype=dtype, rngs=rngs)

    def __call__(self, x, mask=None, deterministic=True):
        for layer in self.layers:
            x = layer(x, mask, deterministic)
        return self.norm(x)


class PartB(nnx.Module):
    def __init__(self, d_model, n_layers, n_heads, n_kv_heads, d_ffn, max_seq_len, use_rope=True, dtype=jnp.float32, remat=False, rngs=None):
        self.layers = nnx.List([
            TransformerBlock(d_model, n_heads, n_kv_heads, d_ffn, max_seq_len, use_rope, dtype, remat, rngs)
            for _ in range(n_layers)
        ])
        self.norm = nnx.LayerNorm(d_model, dtype=dtype, rngs=rngs)

    def __call__(self, x, mask=None, deterministic=True):
        for layer in self.layers:
            x = layer(x, mask, deterministic)
        return self.norm(x)