"""
Configuration Classes for TRM
=============================

This module contains all configuration dataclasses used by TRM.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import json
from pathlib import Path


@dataclass
class TRMConfig:
    """Configuration for Tiny Recursive Model architecture.
    
    The default values match the paper's optimal settings for Sudoku.
    
    Attributes
    ----------
    vocab_size : int
        Number of tokens in vocabulary (e.g., 10 for digits 0-9)
    hidden_dim : int
        Embedding and hidden layer dimension
    num_layers : int
        Number of transformer/mixer layers (paper shows 2 is optimal!)
    num_heads : int
        Number of attention heads (only used if use_attention=True)
    context_length : int
        Maximum sequence length (e.g., 81 for 9x9 Sudoku)
    mlp_ratio : float
        MLP hidden dimension = hidden_dim * mlp_ratio
    dropout : float
        Dropout probability
    n_recursions : int
        Number of latent recursions per step (n in paper)
    T_steps : int
        Number of gradient-free pre-recursion steps (T in paper)
    n_supervision : int
        Maximum number of deep supervision steps
    use_attention : bool
        If True, use multi-head attention; if False, use MLP-Mixer
    
    Examples
    --------
    >>> config = TRMConfig(vocab_size=10, context_length=81)
    >>> config.effective_depth
    42
    """
    
    # Architecture
    vocab_size: int = 10
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    context_length: int = 81
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    
    # Recursion
    n_recursions: int = 6
    T_steps: int = 3
    n_supervision: int = 16
    
    # Architecture choice
    use_attention: bool = False
    
    @property
    def effective_depth(self) -> int:
        """Calculate effective depth per supervision step.
        
        Returns T * (n + 1) * num_layers
        """
        return self.T_steps * (self.n_recursions + 1) * self.num_layers
    
    @property
    def total_depth(self) -> int:
        """Calculate total effective depth across all supervision steps."""
        return self.n_supervision * self.effective_depth
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TRMConfig":
        """Create config from dictionary."""
        return cls(**d)
    
    def save(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "TRMConfig":
        """Load config from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class TrainingConfig:
    """Configuration for training TRM.
    
    Attributes
    ----------
    n_train_samples : int
        Number of training samples (before augmentation)
    n_test_samples : int
        Number of test samples
    n_augmentations : int
        Number of augmentations per sample
    batch_size : int
        Training batch size
    num_epochs : int
        Number of training epochs
    learning_rate : float
        Peak learning rate
    weight_decay : float
        AdamW weight decay
    warmup_steps : int
        Number of learning rate warmup steps
    grad_clip : float
        Gradient clipping norm (0 to disable)
    n_supervision : int
        Supervision steps during training
    n_supervision_eval : int
        Supervision steps during evaluation
    use_act : bool
        Use Adaptive Computation Time for early stopping
    halt_loss_weight : float
        Weight for halting loss (ACT)
    use_ema : bool
        Use Exponential Moving Average of weights
    ema_decay : float
        EMA decay rate
    log_interval : int
        Log metrics every N batches
    eval_interval : int
        Evaluate every N batches
    save_interval : int
        Save checkpoint every N batches
    output_dir : str
        Directory for outputs and checkpoints
    seed : int
        Random seed for reproducibility
    
    Examples
    --------
    >>> config = TrainingConfig(num_epochs=50, batch_size=32)
    >>> print(f"Training for {config.num_epochs} epochs")
    """
    
    # Data
    n_train_samples: int = 1000
    n_test_samples: int = 100
    n_augmentations: int = 100
    
    # Training
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    grad_clip: float = 1.0
    
    # Deep supervision
    n_supervision: int = 16
    n_supervision_eval: int = 16
    use_act: bool = True
    halt_loss_weight: float = 0.5
    
    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999
    
    # Logging
    log_interval: int = 50
    eval_interval: int = 500
    save_interval: int = 1000
    
    # Paths
    output_dir: str = "outputs"
    
    # Reproducibility
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary."""
        return cls(**d)
    
    def save(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load config from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


# Preset configurations for different tasks
SUDOKU_CONFIG = TRMConfig(
    vocab_size=10,
    hidden_dim=256,
    num_layers=2,
    context_length=81,
    n_recursions=6,
    T_steps=3,
    n_supervision=16,
    use_attention=False,  # MLP-Mixer works better for Sudoku
)

MAZE_CONFIG = TRMConfig(
    vocab_size=6,
    hidden_dim=256,
    num_layers=2,
    context_length=900,   # 30x30 grid
    n_recursions=6,
    T_steps=3,
    n_supervision=16,
    use_attention=True,   # Attention better for larger grids
)

ARC_CONFIG = TRMConfig(
    vocab_size=11,        # 0-9 colors + padding
    hidden_dim=512,
    num_layers=2,
    context_length=900,   # 30x30 grid
    n_recursions=6,
    T_steps=3,
    n_supervision=16,
    use_attention=True,
)


def get_preset_config(task: str) -> TRMConfig:
    """Get preset model configuration for a task.
    
    Parameters
    ----------
    task : str
        Task name: 'sudoku', 'maze', or 'arc'
    
    Returns
    -------
    TRMConfig
        Preset configuration for the task
    
    Examples
    --------
    >>> config = get_preset_config('sudoku')
    >>> print(config.context_length)
    81
    """
    presets = {
        "sudoku": SUDOKU_CONFIG,
        "maze": MAZE_CONFIG,
        "arc": ARC_CONFIG,
    }
    
    if task not in presets:
        raise ValueError(
            f"Unknown task: {task}. Available: {list(presets.keys())}"
        )
    
    return presets[task]
