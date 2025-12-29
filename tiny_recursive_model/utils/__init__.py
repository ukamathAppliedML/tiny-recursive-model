"""
Utility Functions
=================

Helper functions for training, checkpointing, and device management.
"""

import torch
import torch.nn as nn
import random
import numpy as np
import copy
from pathlib import Path
from typing import Dict, Any, Optional, Union
import json


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model
    
    Returns
    -------
    int
        Number of trainable parameters
    
    Examples
    --------
    >>> model = TinyRecursiveModel(config)
    >>> print(f"{count_parameters(model):,} parameters")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """Get the best available device.
    
    Priority: CUDA > MPS (Apple Silicon) > CPU
    
    Returns
    -------
    torch.device
        Best available device
    
    Examples
    --------
    >>> device = get_device()
    >>> model = model.to(device)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and GPU)
    
    Parameters
    ----------
    seed : int
        Random seed
    
    Examples
    --------
    >>> set_seed(42)
    >>> # Now results are reproducible
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For MPS, setting torch.manual_seed is sufficient


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    global_step: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Any] = None,
    path: Union[str, Path] = "checkpoint.pt"
) -> None:
    """Save a training checkpoint.
    
    Parameters
    ----------
    model : nn.Module
        Model to save
    optimizer : Optimizer, optional
        Optimizer state
    scheduler : LRScheduler, optional
        Learning rate scheduler state
    epoch : int
        Current epoch
    global_step : int
        Current global step
    metrics : dict, optional
        Training metrics
    config : dataclass, optional
        Model/training configuration
    path : str or Path
        Save path
    
    Examples
    --------
    >>> save_checkpoint(
    ...     model, optimizer, scheduler,
    ...     epoch=10, global_step=5000,
    ...     metrics={"loss": 0.5, "accuracy": 0.95},
    ...     path="outputs/checkpoint.pt"
    ... )
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    if metrics is not None:
        checkpoint["metrics"] = metrics
    
    if config is not None:
        from dataclasses import asdict, is_dataclass
        if is_dataclass(config):
            checkpoint["config"] = asdict(config)
        else:
            checkpoint["config"] = config
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Load a training checkpoint.
    
    Parameters
    ----------
    path : str or Path
        Checkpoint path
    model : nn.Module, optional
        Model to load weights into
    optimizer : Optimizer, optional
        Optimizer to load state into
    scheduler : LRScheduler, optional
        Scheduler to load state into
    device : torch.device, optional
        Device to map tensors to
    
    Returns
    -------
    dict
        Checkpoint data including epoch, step, metrics, config
    
    Examples
    --------
    >>> checkpoint = load_checkpoint("outputs/best.pt", model=model)
    >>> print(f"Loaded from epoch {checkpoint['epoch']}")
    """
    device = device or get_device()
    checkpoint = torch.load(path, map_location=device)
    
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return checkpoint


class EMAModel:
    """Exponential Moving Average of model weights.
    
    Maintains a shadow copy of weights updated as:
        shadow = decay * shadow + (1 - decay) * current
    
    This smooths training noise and often gives better final models.
    
    Parameters
    ----------
    model : nn.Module
        Model to track
    decay : float
        EMA decay rate (0.999 typical)
    
    Examples
    --------
    >>> ema = EMAModel(model, decay=0.999)
    >>> 
    >>> # During training:
    >>> ema.update(model)
    >>> 
    >>> # For evaluation:
    >>> ema.apply_shadow(model)  # Use EMA weights
    >>> evaluate(model)
    >>> ema.restore(model)       # Restore original weights
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        
        for param in self.shadow.parameters():
            param.requires_grad = False
        
        self._backup = None
    
    def update(self, model: nn.Module) -> None:
        """Update shadow weights from model."""
        with torch.no_grad():
            for shadow_param, model_param in zip(
                self.shadow.parameters(), model.parameters()
            ):
                shadow_param.data.mul_(self.decay).add_(
                    model_param.data, alpha=1 - self.decay
                )
    
    def apply_shadow(self, model: nn.Module) -> None:
        """Copy shadow weights to model (saves backup)."""
        self._backup = copy.deepcopy(model.state_dict())
        with torch.no_grad():
            for shadow_param, model_param in zip(
                self.shadow.parameters(), model.parameters()
            ):
                model_param.data.copy_(shadow_param.data)
    
    def restore(self, model: nn.Module) -> None:
        """Restore model weights from backup."""
        if self._backup is not None:
            model.load_state_dict(self._backup)
            self._backup = None


class StableMaxCrossEntropy(nn.Module):
    """Cross-entropy with improved numerical stability.
    
    Uses log_softmax for better numerical behavior.
    Optional temperature scaling for label smoothing.
    
    Parameters
    ----------
    temperature : float
        Temperature for softmax (1.0 = standard)
    
    References
    ----------
    Prieto et al., 2025: "Grokking at the Edge of Numerical Stability"
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : torch.Tensor
            Model outputs [batch, seq_len, vocab_size] or [N, vocab_size]
        targets : torch.Tensor
            Target indices [batch, seq_len] or [N]
        
        Returns
        -------
        torch.Tensor
            Scalar loss
        """
        logits = logits / self.temperature
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Flatten if needed
        if logits.dim() == 3:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            log_probs = log_probs.view(-1, log_probs.size(-1))
        
        # Gather log probs for targets
        target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        
        return -target_log_probs.mean()


def accuracy_score(
    predictions: torch.Tensor, 
    targets: torch.Tensor
) -> float:
    """Calculate element-wise accuracy.
    
    Parameters
    ----------
    predictions : torch.Tensor
        Predictions [batch, seq_len] or [batch, seq_len, vocab_size]
    targets : torch.Tensor
        Targets [batch, seq_len]
    
    Returns
    -------
    float
        Accuracy between 0 and 1
    """
    if predictions.dim() == 3:
        predictions = predictions.argmax(dim=-1)
    return (predictions == targets).float().mean().item()


def puzzle_accuracy(
    predictions: torch.Tensor, 
    targets: torch.Tensor
) -> float:
    """Calculate percentage of fully correct sequences.
    
    Parameters
    ----------
    predictions : torch.Tensor
        Predictions [batch, seq_len]
    targets : torch.Tensor
        Targets [batch, seq_len]
    
    Returns
    -------
    float
        Percentage of sequences that are 100% correct
    """
    if predictions.dim() == 3:
        predictions = predictions.argmax(dim=-1)
    correct_per_sample = (predictions == targets).all(dim=-1).float()
    return correct_per_sample.mean().item()


# Re-export F for convenience
import torch.nn.functional as F
