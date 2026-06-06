"""
Configuration dataclasses for DRWA model and training.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class DRWAConfig:
    """DRWA model architecture configuration.
    
    When N=1, k_max=1, the model is equivalent to a standard dense FFN,
    allowing for fast, stable training. The pool can be expanded later
    via clustering and low-rank decomposition.
    """
    
    # Model dimensions
    vocab_size: int = 50257  # GPT-2 tokenizer
    seq_len: int = 1024
    d_model: int = 768  # Hidden size
    d_ffn: int = 3072  # FFN intermediate size (d_model * 4)
    
    # DRWA-specific parameters
    N: int = 1  # Number of pool vectors (N=1 for dense training)
    k_max: int = 1  # Max retrieved vectors (k_max=1 for dense)
    r: int = 64  # Low-rank dimension (r=64 for N=1 dense, r=16 for N>1)
    S: int = 4  # Number of retrieval aspects (multi-head)
    d_k: int = 256  # Key dimension for retrieval
    d_A: int = 768  # Part A hidden size (same as d_model for simplicity)
    d_B: int = 768  # Part B hidden size
    
    # Reasoning loop (ACC - Adaptive Compute Controller)
    num_loops: int = 3  # Number of reasoning iterations
    halt_threshold: float = 0.99  # Cumulative halt probability threshold
    
    # Retrieval parameters
    lambda_gate: float = 3.0  # Sigmoid gate sharpness
    gamma_init: float = 0.05  # LoRA-style residual scale
    use_ivf: bool = False  # Use IVF clustering for retrieval (N>1 only)
    n_centroids: int = 128  # IVF centroids
    
    # Transformer architecture
    n_layers_A: int = 6  # Layers before assembly
    n_layers_B: int = 6  # Layers after assembly
    n_heads: int = 12
    n_kv_heads: int = 12  # Multi-query attention: set < n_heads
    ffn_mult: int = 4
    use_rope: bool = True  # Rotary positional embeddings
    
    # Precision
    bf16_pool: bool = True  # Store pool in bfloat16
    compute_dtype: Optional[str] = None  # None=float32, "bfloat16" for compute
    
    # Gradient checkpointing
    remat: bool = False  # Enable gradient checkpointing
    
    # Misc
    dropout: float = 0.0
    
    def __post_init__(self):
        """Validate configuration and set derived fields."""
        # Compute d_B if not set
        if self.d_B == 0:
            self.d_B = self.d_model
        
        # Validate dimensions
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.d_A <= self.d_model, "d_A should be <= d_model"
        
        # For N=1 (dense training), ensure k_max=1 and use full rank
        if self.N == 1:
            self.k_max = 1
            # Full rank for dense mode
            if self.r == 64:
                # Compute effective FFN dimension
                pool_dim = 2 * self.d_B * self.r + self.d_B
                assert pool_dim >= self.d_ffn, f"Pool dimension {pool_dim} too small for FFN {self.d_ffn}"
    
    @classmethod
    def dense_small(cls) -> "DRWAConfig":
        """Small dense configuration for testing."""
        return cls(
            vocab_size=50257,
            seq_len=512,
            d_model=256,
            d_ffn=1024,
            d_A=256,
            d_B=256,
            N=1,
            k_max=1,
            r=64,
            n_layers_A=2,
            n_layers_B=2,
            n_heads=4,
            n_kv_heads=4,
        )
    
    @classmethod
    def dense_medium(cls) -> "DRWAConfig":
        """Medium dense configuration (125M params)."""
        return cls(
            vocab_size=50257,
            seq_len=1024,
            d_model=768,
            d_ffn=3072,
            d_A=768,
            d_B=768,
            N=1,
            k_max=1,
            r=64,
            n_layers_A=6,
            n_layers_B=6,
            n_heads=12,
        )
    
    @classmethod
    def dense_large(cls) -> "DRWAConfig":
        """Large dense configuration (350M params)."""
        return cls(
            vocab_size=50257,
            seq_len=2048,
            d_model=1024,
            d_ffn=4096,
            d_A=1024,
            d_B=1024,
            N=1,
            k_max=1,
            r=64,
            n_layers_A=12,
            n_layers_B=12,
            n_heads=16,
            n_kv_heads=16,
            bf16_pool=True,
            compute_dtype="bfloat16",
        )

    @classmethod
    def dense_colab(cls) -> "DRWAConfig":
        """Colab TPU v5e-1 (~130M params, fits 16GB HBM with bf16+remat)."""
        return cls(
            vocab_size=50257,
            seq_len=1024,
            d_model=768,
            d_ffn=3072,
            d_A=768,
            d_B=768,
            N=1,
            k_max=1,
            r=64,
            n_layers_A=4,
            n_layers_B=4,
            n_heads=12,
            n_kv_heads=6,
            bf16_pool=True,
            compute_dtype="bfloat16",
            remat=True,
        )

    @classmethod
    def dense_xl(cls) -> "DRWAConfig":
        """3B dense configuration for TPU v5e-8 with model parallelism."""
        return cls(
            vocab_size=50257,
            seq_len=2048,
            d_model=2560,
            d_ffn=10240,
            d_A=2560,
            d_B=2560,
            N=1,
            k_max=1,
            r=64,
            S=4,
            d_k=256,
            n_layers_A=20,
            n_layers_B=20,
            n_heads=40,
            n_kv_heads=8,  # GQA ratio 5:1
            bf16_pool=True,
            compute_dtype="bfloat16",
            remat=True,
        )
    
    @classmethod
    def drwa_expanded(cls, N: int = 1024) -> "DRWAConfig":
        """DRWA expanded configuration after pool expansion."""
        return cls(
            vocab_size=50257,
            seq_len=1024,
            d_model=768,
            d_ffn=3072,
            N=N,  # Expanded pool
            k_max=16,  # Retrieve top-16
            r=16,  # Low-rank
            n_layers_A=6,
            n_layers_B=6,
            n_heads=12,
        )
    
    @property
    def pool_dim(self) -> int:
        """Dimension of each pool vector."""
        return 2 * self.d_B * self.r + self.d_B
    
    @property
    def total_params(self) -> int:
        """Approximate total parameter count including GQA."""
        head_dim = self.d_model // self.n_heads
        d_kv = self.n_kv_heads * head_dim

        # Embedding
        embed_params = self.vocab_size * self.d_model

        # Per transformer block: attention + FFN + norms + biases
        # Attention: wq(d,d) + wk(d,d_kv) + wv(d,d_kv) + wo(d,d)
        attn_params = 2 * self.d_model ** 2 + 2 * self.d_model * d_kv
        # Add biases for all 4 projections
        attn_params += self.d_model + d_kv + d_kv + self.d_model
        # FFN: w1(d,d_ffn) + w2(d_ffn,d) + biases
        ffn_params = self.d_model * self.d_ffn + self.d_ffn * self.d_model + self.d_ffn + self.d_model
        # LayerNorms: 2 per block, each with weight + bias
        norm_params = 4 * self.d_model
        layer_params = attn_params + ffn_params + norm_params

        n_layers = self.n_layers_A + self.n_layers_B
        transformer_params = n_layers * layer_params

        # Part A / Part B final norms
        final_norm_params = 4 * self.d_model

        # Assembly (N=1): W_base(d_B,d_A) + b_base(d_B) + U(d_B,r) + V(r,d_A) + b(d_B) + gamma(1) + norm(2*d_B)
        assembly_params = self.d_B * self.d_A + self.d_B + self.d_B * self.r + self.r * self.d_A + self.d_B + 1 + 2 * self.d_B
        if self.d_A != self.d_B:
            assembly_params += self.d_A * self.d_B

        # LM head
        lm_head_params = self.d_B * self.vocab_size + self.vocab_size

        return embed_params + transformer_params + final_norm_params + assembly_params + lm_head_params


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    
    # Training schedule
    total_steps: int = 100_000
    steps_per_window: int = 1  # Steps per JIT compilation (1=single-step, 32+=windowed)
    batch_size: int = 4
    
    # Learning rate
    lr_pool: float = 3e-5
    lr_transformer: float = 1e-4
    lr_retrieval: float = 1e-4
    lr_reasoning: float = 1e-4  # ACC parameters (tau, gamma)
    
    # LR schedule
    warmup_steps: int = 1_000
    lr_warmup_steps: int = 500
    lr_min_scale: float = 0.1
    
    # Phase schedule (for DRWA)
    gate_on_steps: int = 10_000  # When to enable sigmoid gate
    gate_ramp_steps: int = 5_000  # Gate ramp duration
    
    # Gradient
    grad_clip_norm: float = 1.0
    
    # Pool EMA (for collapse detection)
    ema_decay: float = 0.999
    ema_threshold: float = 1e-5
    
    # Safety
    nan_emergency_stop: int = 5
    loss_spike_sigma: float = 5.0
    revival_interval_steps: int = 1000
    
    # Checkpointing
    checkpoint_every: int = 5000
    keep_checkpoints: int = 3
    
    # Validation
    val_every: int = 500
    val_batches: int = 32
    
    # Misc
    seed: int = 42
    log_every: int = 10  # Log every N windows