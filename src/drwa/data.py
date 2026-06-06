"""
Data loaders for DRWA training.
"""

import jax
import jax.numpy as jnp
import numpy as np
from threading import Thread
from typing import Optional, Iterator, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer
from flax import nnx


class DataLoader:
    """Base class for data loaders."""
    
    def __init__(
        self,
        seq_len: int,
        batch_size: int,
        vocab_size: int,
        seed: int = 42,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)
    
    def get_window(self, steps: int) -> jnp.ndarray:
        """Get a window of steps.
        
        Returns:
            [steps, batch_size, seq_len] int32 array
        """
        raise NotImplementedError
    
    def reset(self):
        """Reset loader state."""
        pass


class SyntheticDataLoader(DataLoader):
    """Synthetic data for testing and benchmarking."""
    
    def __init__(
        self,
        seq_len: int,
        batch_size: int,
        vocab_size: int,
        seed: int = 42,
        pattern: str = "random",
    ):
        super().__init__(seq_len, batch_size, vocab_size, seed)
        self.pattern = pattern
    
    def get_window(self, steps: int) -> jnp.ndarray:
        """Generate synthetic data window."""
        key, self.rng = jax.random.split(self.rng)
        
        if self.pattern == "random":
            # Random token IDs
            data = jax.random.randint(
                key,
                (steps, self.batch_size, self.seq_len),
                0,
                self.vocab_size,
                dtype=jnp.int32,
            )
        elif self.pattern == "repeat":
            # Repeating pattern for testing
            base = jnp.arange(self.seq_len) % 100 + 1
            data = jnp.tile(base, (steps, self.batch_size, 1))
            data = data.astype(jnp.int32)
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        
        return data


class HuggingFaceDataLoader(DataLoader):
    """HuggingFace datasets streaming loader."""
    
    def __init__(
        self,
        seq_len: int,
        batch_size: int,
        vocab_size: int,
        hf_path: str,
        hf_subset: Optional[str] = None,
        hf_text_column: str = "text",
        tokenizer_name: str = "gpt2",
        seed: int = 42,
        split: str = "train",
    ):
        super().__init__(seq_len, batch_size, vocab_size, seed)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.eos_token_id = self.tokenizer.eos_token_id
        
        # Load dataset in streaming mode
        self.dataset = load_dataset(
            hf_path,
            hf_subset,
            split=split,
            streaming=True,
        )
        
        # Buffer for packed sequences
        self.buffer = np.array([], dtype=np.int32)
        self.buffer_pos = 0
        
        # Text column
        self.text_column = hf_text_column
        self.dataset_iter = iter(self.dataset)
    
    def _refill_buffer(self, needed: int):
        """Refill buffer with at least `needed` tokens."""
        while len(self.buffer) - self.buffer_pos < needed:
            try:
                # Get next example
                example = next(self.dataset_iter)
                text = example[self.text_column]
                
                # Tokenize
                tokens = self.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                    return_tensors="np",
                )[0]
                
                # Wrap with EOS
                wrapped = np.concatenate([
                    [self.eos_token_id],
                    tokens,
                    [self.eos_token_id],
                ])
                
                # Append to buffer
                self.buffer = np.concatenate([self.buffer[self.buffer_pos:], wrapped])
                self.buffer_pos = 0
                
            except StopIteration:
                # Dataset exhausted, restart
                self.dataset = load_dataset(
                    self.dataset.builder_name,
                    self.dataset.config_name,
                    split=self.dataset.split,
                    streaming=True,
                )
                self.dataset_iter = iter(self.dataset)
    
    def get_window(self, steps: int) -> np.ndarray:
        """Get a window of packed token sequences."""
        needed = steps * self.batch_size * self.seq_len
        self._refill_buffer(needed)
        
        # Extract window
        window_flat = self.buffer[self.buffer_pos:self.buffer_pos + needed]
        self.buffer_pos += needed
        
        # Reshape
        window = window_flat.reshape(steps, self.batch_size, self.seq_len)
        
        return window.astype(np.int32)
    
    def reset(self):
        """Reset buffer."""
        self.buffer = np.array([], dtype=np.int32)
        self.buffer_pos = 0


class TinyStoriesDataLoader(HuggingFaceDataLoader):
    """TinyStories dataset loader."""
    
    def __init__(
        self,
        seq_len: int,
        batch_size: int,
        vocab_size: int,
        tokenizer_name: str = "gpt2",
        seed: int = 42,
    ):
        super().__init__(
            seq_len=seq_len,
            batch_size=batch_size,
            vocab_size=vocab_size,
            hf_path="roneneldan/TinyStories",
            hf_subset=None,
            hf_text_column="text",
            tokenizer_name=tokenizer_name,
            seed=seed,
            split="train",
        )


class PrefetchDataLoader:
    """Wraps a DataLoader to prefetch next window in a background thread."""

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self._prefetch_thread = None
        self._prefetch_result = None
        self._prefetch_error = None

    def get_window(self, steps: int) -> np.ndarray:
        if self._prefetch_thread is not None:
            self._prefetch_thread.join()
            if self._prefetch_error is not None:
                raise self._prefetch_error
            result = self._prefetch_result
        else:
            result = self.loader.get_window(steps)

        self._prefetch_error = None
        self._prefetch_result = None

        def _prefetch():
            try:
                self._prefetch_result = self.loader.get_window(steps)
            except Exception as e:
                self._prefetch_error = e
        self._prefetch_thread = Thread(target=_prefetch, daemon=True)
        self._prefetch_thread.start()

        return result

    def reset(self):
        if self._prefetch_thread is not None:
            self._prefetch_thread.join()
            self._prefetch_thread = None
        self.loader.reset()


def create_data_loader(
    config,
    train_config,
    process_index: int = 0,
    process_count: int = 1,
    prefetch: bool = True,
) -> DataLoader:
    """Create data loader based on config.

    Args:
        config: DRWAConfig
        train_config: TrainConfig
        process_index: Process index for multi-host (unused for random)
        process_count: Total processes for multi-host (unused for random)
        prefetch: Wrap loader in background thread prefetch

    Returns:
        DataLoader instance (wrapped in PrefetchDataLoader if prefetch=True)
    """
    data_cfg = config.get("data", {})
    source = data_cfg.get("source", "random")
    
    if source == "random":
        loader = SyntheticDataLoader(
            seq_len=config["model"]["seq_len"],
            batch_size=train_config.batch_size // process_count,
            vocab_size=config["model"]["vocab_size"],
            seed=train_config.seed + process_index,
            pattern="random",
        )
    
    elif source == "tiny_stories":
        loader = TinyStoriesDataLoader(
            seq_len=config["model"]["seq_len"],
            batch_size=train_config.batch_size // process_count,
            vocab_size=config["model"]["vocab_size"],
            tokenizer_name=data_cfg.get("hf_tokenizer", "gpt2"),
            seed=train_config.seed + process_index,
        )
    
    elif source == "hf":
        loader = HuggingFaceDataLoader(
            seq_len=config["model"]["seq_len"],
            batch_size=train_config.batch_size // process_count,
            vocab_size=config["model"]["vocab_size"],
            hf_path=data_cfg["hf_path"],
            hf_subset=data_cfg.get("hf_subset"),
            hf_text_column=data_cfg.get("hf_text_column", "text"),
            tokenizer_name=data_cfg.get("hf_tokenizer", "gpt2"),
            seed=train_config.seed + process_index,
        )
    
    else:
        raise ValueError(f"Unknown data source: {source}")
    
    return PrefetchDataLoader(loader) if prefetch else loader