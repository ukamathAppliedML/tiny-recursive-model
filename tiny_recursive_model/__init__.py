"""
Tiny Recursive Model (TRM)
==========================

A minimal implementation of recursive reasoning networks from
"Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871).

Quick Start
-----------
>>> from tiny_recursive_model import TinyRecursiveModel, TRMConfig
>>> 
>>> config = TRMConfig(vocab_size=10, context_length=81)
>>> model = TinyRecursiveModel(config)
>>> 
>>> # Forward pass
>>> x = torch.randint(0, 10, (4, 81))
>>> (y, z), logits, halt = model(x)

For training, see :func:`train` or the CLI::

    trm train --task sudoku --epochs 50

"""

__version__ = "0.1.0"
__author__ = "TRM Contributors"

from .config import TRMConfig, TrainingConfig
from .model import TinyRecursiveModel
from .trainer import train, Trainer
from .utils import count_parameters, get_device, set_seed

__all__ = [
    # Config
    "TRMConfig",
    "TrainingConfig",
    # Model
    "TinyRecursiveModel",
    # Training
    "train",
    "Trainer",
    # Utils
    "count_parameters",
    "get_device",
    "set_seed",
]
