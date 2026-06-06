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


def _psum_if_parallel(x, axis_name):
    """All-reduce x along axis_name when inside shard_map; no-op otherwise.

    Under shard_map each device holds a partial result and the psum combines
    them.  During eager generation (outside any shard_map context) JAX's
    transparent sharding already exposes the full logical tensor, so the
    operation is correctly skipped.
    """
    try:
        return jax.lax.psum(x, axis_name)
    except NameError:
        return x


class CausalSelfAttention(nnx.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, max_seq_len, use_rope=True, dtype=jnp.float32, rngs=None, model_axis=None):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope
        self.dtype = dtype
        self.model_axis = model_axis
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
        # Infer per-device head counts from the actual projection output size.
        # Inside shard_map the column-parallel kernel is sliced, giving
        # n_heads/n_model heads per device.  Outside shard_map (generation)
        # JAX exposes the full logical weight, giving all n_heads.
        q_proj = self.wq(x)
        k_proj = self.wk(x)
        v_proj = self.wv(x)
        n_h  = q_proj.shape[-1] // self.head_dim
        n_kv = k_proj.shape[-1] // self.head_dim
        q = q_proj.reshape(B, T, n_h,  self.head_dim)
        k = k_proj.reshape(B, T, n_kv, self.head_dim)
        v = v_proj.reshape(B, T, n_kv, self.head_dim)
        if self.use_rope:
            q = apply_rope(q, self.cos.value, self.sin.value)
            k = apply_rope(k, self.cos.value, self.sin.value)
        out = jax.nn.dot_product_attention(q, k, v, is_causal=True, mask=mask)
        out = out.reshape(B, T, n_h * self.head_dim)
        out = self.wo(out)
        if self.model_axis is not None:
            out = _psum_if_parallel(out, self.model_axis)
        return out


class FFN(nnx.Module):
    def __init__(self, d_model, d_ffn, dtype=jnp.float32, rngs=None, model_axis=None):
        self.w1 = nnx.Linear(d_model, d_ffn, dtype=dtype, rngs=rngs)
        self.w2 = nnx.Linear(d_ffn, d_model, dtype=dtype, rngs=rngs)
        self.dtype = dtype
        self.model_axis = model_axis

    def __call__(self, x):
        h = self.w2(jax.nn.gelu(self.w1(x)))
        if self.model_axis is not None:
            h = _psum_if_parallel(h, self.model_axis)
        return h


class TransformerBlock(nnx.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, d_ffn, max_seq_len, use_rope=True, dtype=jnp.float32, remat=False, rngs=None, model_axis=None):
        self.norm1 = nnx.LayerNorm(d_model, dtype=dtype, rngs=rngs)
        self.norm2 = nnx.LayerNorm(d_model, dtype=dtype, rngs=rngs)
        self.attn = CausalSelfAttention(d_model, n_heads, n_kv_heads, max_seq_len, use_rope, dtype, rngs, model_axis=model_axis)
        self.ffn = FFN(d_model, d_ffn, dtype, rngs, model_axis=model_axis)
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


def _scan_layers(layers, x, mask, deterministic, remat):
    """Run a homogeneous nnx.List of TransformerBlocks via jax.lax.scan.

    Stacks each layer's parameters along a leading axis at trace time, then
    scans over them.  This collapses N unrolled layer ops in the XLA graph
    down to a single scan body, which lets XLA pipeline compute with the
    all-reduce collectives and dramatically shrinks compilation time for deep
    models.

    Remat (gradient checkpointing) is applied to the scan body rather than
    per-block, so the granularity and memory savings are identical.
    """
    # graphdef captures the static module structure (same for every layer)
    graphdef, _ = nnx.split(layers[0])

    # Stack each layer's live parameters into [n_layers, *param_shape] arrays
    layer_states = [nnx.split(l)[1] for l in layers]
    stacked_state = jax.tree_util.tree_map(
        lambda *leaves: jnp.stack(leaves, axis=0), *layer_states
    )

    def layer_step(carry, state_i):
        layer = nnx.merge(graphdef, state_i)
        return layer(carry, mask, deterministic), None

    if remat:
        layer_step = jax.checkpoint(layer_step)

    x, _ = jax.lax.scan(layer_step, x, stacked_state)
    return x


class PartA(nnx.Module):
    def __init__(self, d_model, n_layers, n_heads, n_kv_heads, d_ffn, max_seq_len, use_rope=True, dtype=jnp.float32, remat=False, rngs=None, model_axis=None):
        self._remat = remat
        # remat is handled at the scan-body level; individual blocks run plain
        self.layers = nnx.List([
            TransformerBlock(d_model, n_heads, n_kv_heads, d_ffn, max_seq_len, use_rope, dtype, remat=False, rngs=rngs, model_axis=model_axis)
            for _ in range(n_layers)
        ])
        self.norm = nnx.LayerNorm(d_model, dtype=dtype, rngs=rngs)

    def __call__(self, x, mask=None, deterministic=True):
        x = _scan_layers(self.layers, x, mask, deterministic, self._remat)
        return self.norm(x)


class PartB(nnx.Module):
    def __init__(self, d_model, n_layers, n_heads, n_kv_heads, d_ffn, max_seq_len, use_rope=True, dtype=jnp.float32, remat=False, rngs=None, model_axis=None):
        self._remat = remat
        # remat is handled at the scan-body level; individual blocks run plain
        self.layers = nnx.List([
            TransformerBlock(d_model, n_heads, n_kv_heads, d_ffn, max_seq_len, use_rope, dtype, remat=False, rngs=rngs, model_axis=model_axis)
            for _ in range(n_layers)
        ])
        self.norm = nnx.LayerNorm(d_model, dtype=dtype, rngs=rngs)

    def __call__(self, x, mask=None, deterministic=True):
        x = _scan_layers(self.layers, x, mask, deterministic, self._remat)
        return self.norm(x)
