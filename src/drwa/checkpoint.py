"""
Checkpointing utilities for DRWA training.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from pathlib import Path
from typing import Dict, Any, Optional
import orbax.checkpoint as ocp
import numpy as np
from datetime import datetime


class CheckpointManager:
    """Checkpoint manager with Orbax."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_to_keep: int = 3,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        
        # Orbax checkpointer
        self.checkpointer = ocp.StandardCheckpointer()
    
    def save(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        step: int,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """Save checkpoint.
        
        Args:
            model: DRWA model
            optimizer: Optimizer state
            step: Current training step
            metrics: Optional metrics to save
        """
        # Extract state
        model_state = nnx.state(model, nnx.Param)
        opt_state = optimizer.opt_state
        
        # Convert to numpy for saving
        model_np = jax.tree_util.tree_map(np.array, model_state)
        opt_np = jax.tree_util.tree_map(
            lambda x: np.array(x) if hasattr(x, 'shape') else x,
            opt_state
        )
        
        # Create checkpoint path
        checkpoint_path = self.checkpoint_dir / f"step_{step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = checkpoint_path / "model"
        self.checkpointer.save(model_path, ocp.args.PyTreeSave(model_np))
        
        # Save optimizer
        opt_path = checkpoint_path / "optimizer"
        self._save_optimizer_state(opt_path, opt_np)
        
        # Save metadata
        metadata = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics or {},
        }
        np.save(checkpoint_path / "metadata.npy", metadata, allow_pickle=True)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        print(f"✓ Checkpoint saved: {checkpoint_path}")
    
    def load(
        self,
        model: nnx.Module,
        optimizer: Optional[nnx.Optimizer] = None,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Load checkpoint.
        
        Args:
            model: Model to load into
            optimizer: Optional optimizer to load into
            step: Specific step to load (if None, load latest)
        
        Returns:
            Metadata dict with step and metrics
        """
        # Find checkpoint
        if step is None:
            checkpoint_path = self._get_latest_checkpoint()
        else:
            checkpoint_path = self.checkpoint_dir / f"step_{step}"
        
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load model
        model_path = checkpoint_path / "model"
        model_state = self.checkpointer.restore(model_path)
        
        # Apply to model
        model_state_pytree = jax.tree_util.tree_map(
            lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x,
            model_state
        )
        nnx.update(model, model_state_pytree)
        
        # Load optimizer if provided
        if optimizer is not None:
            opt_path = checkpoint_path / "optimizer"
            opt_state = self._load_optimizer_state(opt_path)
            optimizer.opt_state = opt_state
        
        # Load metadata
        metadata_path = checkpoint_path / "metadata.npy"
        if metadata_path.exists():
            metadata = np.load(metadata_path, allow_pickle=True).item()
        else:
            metadata = {"step": step, "metrics": {}}
        
        print(f"✓ Checkpoint loaded: {checkpoint_path}")
        return metadata
    
    def _get_latest_checkpoint(self) -> Path:
        """Get latest checkpoint path."""
        checkpoints = sorted(self.checkpoint_dir.glob("step_*"))
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {self.checkpoint_dir}")
        return checkpoints[-1]
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints."""
        checkpoints = sorted(self.checkpoint_dir.glob("step_*"))
        while len(checkpoints) > self.max_to_keep:
            old_ckpt = checkpoints.pop(0)
            import shutil
            shutil.rmtree(old_ckpt)
    
    def _save_optimizer_state(self, path: Path, state):
        """Save optimizer state."""
        path.mkdir(parents=True, exist_ok=True)
        # Save as pickled file for simplicity
        import pickle
        with open(path / "state.pkl", "wb") as f:
            pickle.dump(state, f)
    
    def _load_optimizer_state(self, path: Path):
        """Load optimizer state."""
        import pickle
        with open(path / "state.pkl", "rb") as f:
            return pickle.load(f)


class MetricsLogger:
    """Metrics logger for training tracking."""
    
    def __init__(
        self,
        log_dir: str,
        use_wandb: bool = False,
        wandb_project: str = "drwa",
        wandb_name: Optional[str] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        self.metrics_history = []
        
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project,
                    name=wandb_name,
                    dir=str(self.log_dir),
                )
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not installed. Disabling wandb logging.")
                self.use_wandb = False
    
    def log(self, metrics: Dict[str, Any], step: int):
        """Log metrics.
        
        Args:
            metrics: Dict of metric name -> value
            step: Current training step
        """
        # Add step to metrics
        metrics_with_step = {**metrics, "step": step}
        
        # Store locally
        self.metrics_history.append(metrics_with_step)
        
        # Log to wandb
        if self.use_wandb:
            self.wandb.log(metrics, step=step)
    
    def save(self, filename: str = "metrics.npy"):
        """Save metrics history to file."""
        np.save(
            self.log_dir / filename,
            self.metrics_history,
            allow_pickle=True,
        )
    
    def close(self):
        """Close wandb run if active."""
        if self.use_wandb:
            self.wandb.finish()