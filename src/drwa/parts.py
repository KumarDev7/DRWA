"""
Transformer building blocks: Attention, FFN, TransformerBlock.
"""

import jax
import jax.numpy as jnp
from flax import nnx


def precompute_rope_freqs(dim: int, max_seq_len: int, base: float = 10000.0):
    """Precompute RoPE frequencies."""
    freqs = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    positions = jnp.arange(max_seq_len, dtype=jnp.float32)
    angles = positions[:, None] * freqs[None, :]
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(x, cos, sin):
    """Apply rotary positional embeddings."""
    # x: [B, T, n_heads, head_dim]
    B, T, n_heads, head_dim = x.shape
    assert head_dim % 2 == 0
    
    x1 = x[..., :head_dim//2]
    x2 = x[..., head_dim//2:]
    
    cos = cos[:T, :][None, :, None, :]  # [1, T, 1, head_dim//2]
    sin = sin[:T, :][None, :, None, :]
    
    rotated = jnp.concatenate([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], axis=-1)
    
    return rotated


class CausalSelfAttention(nnx.Module):
    """Multi-head causal self-attention with optional GQA and RoPE."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        use_rope: bool = True,
        dtype=jnp.float32,
        rngs: nnx.Rngs = None,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope
        self.dtype = dtype
        
        # Projections
        self.wq = nnx.Linear(d_model, n_heads * self.head_dim, dtype=dtype, rngs=rngs)
        self.wk = nnx.Linear(d_model, n_kv_heads * self.head_dim, dtype=dtype, rngs=rngs)
        self.wv = nnx.Linear(d_model, n_kv_heads * self.head_dim, dtype=dtype, rngs=rngs)
        self.wo = nnx.Linear(n_heads * self.head_dim, d_model, dtype=dtype, rngs=rngs)
        
        # RoPE frequencies (cached)
        if use_rope:
            cos, sin = precompute_rope_freqs(self.head_dim, max_seq_len)
            self.cos = nnx.Variable(cos)
            self.sin = nnx.Variable(sin)
    
    def __call__(self, x, mask=None, deterministic=True):
        B, T, d = x.shape
        
        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).reshape(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).reshape(B, T, self.n_kv_heads, self.head_dim)
        
        if self.use_rope:
            q = apply_rope(q, self.cos.value, self.sin.value)
            k = apply_rope(k, self.cos.value, self.sin.value)
        
        if self.n_kv_heads < self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            k = jnp.repeat(k, n_rep, axis=2)
            v = jnp.repeat(v, n_rep, axis=2)
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = jnp.einsum('bthd,bshd->bhts', q, k) * scale
        
        # Causal mask
        if mask is None:
            mask = jnp.triu(jnp.full((T, T), -1e9), k=1)
        attn = attn + mask[None, None, :, :]
        
        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(self.dtype)
        
        # Attend to values
        out = jnp.einsum('bhts,bshd->bthd', attn, v)
        out = out.reshape(B, T, self.n_heads * self.head_dim)
        
        return self.wo(out)


class FFN(nnx.Module):
    """Standard FFN with GELU activation."""
    
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        dtype=jnp.float32,
        rngs: nnx.Rngs = None,
    ):
        self.w1 = nnx.Linear(d_model, d_ffn, dtype=dtype, rngs=rngs)
        self.w2 = nnx.Linear(d_ffn, d_model, dtype=dtype, rngs=rngs)
        self.dtype = dtype
    
    def __call__(self, x):
        return self.w2(jax.nn.gelu(self.w1(x)))


class TransformerBlock(nnx.Module):
    """Transformer block with pre-norm."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_ffn: int,
        max_seq_len: int,
        use_rope: bool = True,
        dtype=jnp.float32,
        rngs: nnx.Rngs = None,
    ):
        self.norm1 = nnx.LayerNorm(d_model, dtype=dtype, rngs=rngs)
        self.norm2 = nnx.LayerNorm(d_model, dtype=dtype, rngs=rngs)
        
        self.attn = CausalSelfAttention(
            d_model, n_heads, n_kv_heads, max_seq_len, use_rope, dtype, rngs
        )
        self.ffn = FFN(d_model, d_ffn, dtype, rngs)
    
    def __call__(self, x, mask=None, deterministic=True):
        # Pre-norm attention
        x = x + self.attn(self.norm1(x), mask, deterministic)
        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


class PartA(nnx.Module):
    """Part A: Transformer stack before assembly."""
    
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        d_ffn: int,
        max_seq_len: int,
        use_rope: bool = True,
        dtype=jnp.float32,
        rngs: nnx.Rngs = None,
    ):
        self.layers = nnx.List([
            TransformerBlock(d_model, n_heads, n_kv_heads, d_ffn, max_seq_len, use_rope, dtype, rngs)
            for _ in range(n_layers)
        ])
        self.norm = nnx.LayerNorm(d_model, dtype=dtype, rngs=rngs)
    
    def __call__(self, x, mask=None, deterministic=True):
        for layer in self.layers:
            x = layer(x, mask, deterministic)
        return self.norm(x)


class PartB(nnx.Module):
    """Part B: Transformer stack after assembly."""
    
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        d_ffn: int,
        max_seq_len: int,
        use_rope: bool = True,
        dtype=jnp.float32,
        rngs: nnx.Rngs = None,
    ):
        self.layers = nnx.List([
            TransformerBlock(d_model, n_heads, n_kv_heads, d_ffn, max_seq_len, use_rope, dtype, rngs)
            for _ in range(n_layers)
        ])
        self.norm = nnx.LayerNorm(d_model, dtype=dtype, rngs=rngs)
    
    def __call__(self, x, mask=None, deterministic=True):
        for layer in self.layers:
            x = layer(x, mask, deterministic)
        return self.norm(x)