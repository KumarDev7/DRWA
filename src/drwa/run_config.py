"""
Run configuration with YAML support.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path
import yaml
import json

from .config import DRWAConfig, TrainConfig


@dataclass
class ShardingConfig:
    """Mesh sharding configuration."""
    n_data: int = 1  # Data parallel replicas
    n_model: int = 1  # Model parallel replicas (for pool)


@dataclass
class DataConfig:
    """Data source configuration."""
    source: str = "random"  # "random", "pattern", "tiny_stories", "hf"
    hf_path: Optional[str] = None
    hf_subset: Optional[str] = None
    hf_text_column: str = "text"
    hf_tokenizer: str = "gpt2"
    seq_len: int = 1024
    val_hf_path: Optional[str] = None
    val_hf_subset: Optional[str] = None
    val_hf_text_column: Optional[str] = None


@dataclass
class CheckpointConfig:
    """Checkpointing configuration."""
    dir: str = "checkpoints"
    every: int = 5000
    resume: Optional[str] = None  # Path to checkpoint to resume from
    keep: int = 3


@dataclass
class WandbConfig:
    """Weights & Biases logging configuration."""
    project: str = "drwa"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    log_every: int = 10


@dataclass
class GenerateConfig:
    """Periodic generation / sampling configuration."""
    every: int = 0              # Generate every N steps; 0 = disabled
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.9
    prompts: List[str] = field(default_factory=lambda: ["Once upon a time"])
    seed: int = 42


@dataclass
class RunConfig:
    """Complete run configuration."""
    model: DRWAConfig
    train: TrainConfig
    sharding: ShardingConfig = field(default_factory=ShardingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        result = {}
        for key, value in asdict(self).items():
            if hasattr(value, '__dataclass_fields__'):
                result[key] = asdict(value)
            else:
                result[key] = value
        return result
    
    def save(self, path: str) -> None:
        """Save configuration to YAML file."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path_obj, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str) -> "RunConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        
        # Parse nested configs
        model = DRWAConfig(**data.get('model', {}))
        train = TrainConfig(**data.get('train', {}))
        sharding = ShardingConfig(**data.get('sharding', {}))
        data_cfg = DataConfig(**data.get('data', {}))
        checkpoint = CheckpointConfig(**data.get('checkpoint', {}))
        wandb = WandbConfig(**data.get('wandb', {}))
        generate = GenerateConfig(**data.get('generate', {}))

        return cls(
            model=model,
            train=train,
            sharding=sharding,
            data=data_cfg,
            checkpoint=checkpoint,
            wandb=wandb,
            generate=generate,
        )
    
    @classmethod
    def from_preset(cls, preset: str) -> "RunConfig":
        """Create configuration from preset name.
        
        Presets:
        - "dense_small": Small dense (test)
        - "dense_medium": Medium dense (125M)
        - "dense_large": Large dense (350M)
        - "dense_xl": XL dense (3B, TPU v5e-8)
        - "drwa_1b": DRWA expanded 1B
        - "drwa_3b": DRWA expanded 3B
        """
        presets = {
            "dense_small": DRWAConfig.dense_small,
            "dense_medium": DRWAConfig.dense_medium,
            "dense_large": DRWAConfig.dense_large,
            "dense_xl": DRWAConfig.dense_xl,
            "dense_colab": DRWAConfig.dense_colab,
            "drwa_1b": lambda: DRWAConfig.drwa_expanded(N=1024),
            "drwa_3b": lambda: DRWAConfig.drwa_expanded(N=2048),
        }
        
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
        
        return cls(
            model=presets[preset](),
            train=TrainConfig(),
        )


def load_config(path_or_preset: str) -> RunConfig:
    """Load configuration from YAML file or preset name.
    
    Args:
        path_or_preset: Path to YAML file or preset name
    
    Returns:
        RunConfig instance
    """
    path_obj = Path(path_or_preset)
    
    if path_obj.exists():
        return RunConfig.load(path_or_preset)
    elif path_or_preset in ["dense_small", "dense_medium", "dense_large", "dense_xl", "dense_colab", "drwa_1b", "drwa_3b"]:
        return RunConfig.from_preset(path_or_preset)
    else:
        raise ValueError(f"Config not found: {path_or_preset}. "
                        f"Provide a YAML path or preset: dense_small, dense_medium, dense_large, dense_xl, drwa_1b, drwa_3b")


def save_config(config: RunConfig, path: str) -> None:
    """Save configuration to YAML file."""
    config.save(path)