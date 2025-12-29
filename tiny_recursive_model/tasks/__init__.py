"""
Task Registry
=============

Registry for task datasets and configurations.
Allows easy extension to new tasks.
"""

from typing import Callable, Dict, Any, Tuple
from torch.utils.data import Dataset

# Global registry
_TASK_REGISTRY: Dict[str, Callable] = {}


def register_task(name: str):
    """Decorator to register a task factory function.
    
    Parameters
    ----------
    name : str
        Task name for CLI and API
    
    Examples
    --------
    >>> @register_task("my_task")
    ... def create_my_task(config):
    ...     train_ds = MyDataset(...)
    ...     test_ds = MyDataset(...)
    ...     return train_ds, test_ds
    """
    def decorator(func: Callable) -> Callable:
        _TASK_REGISTRY[name] = func
        return func
    return decorator


def get_task(name: str) -> Callable:
    """Get a registered task factory.
    
    Parameters
    ----------
    name : str
        Task name
    
    Returns
    -------
    Callable
        Factory function that returns (train_dataset, test_dataset)
    """
    if name not in _TASK_REGISTRY:
        raise ValueError(
            f"Unknown task: {name}. "
            f"Available: {list(_TASK_REGISTRY.keys())}"
        )
    return _TASK_REGISTRY[name]


def list_tasks() -> list:
    """List all registered tasks."""
    return list(_TASK_REGISTRY.keys())


# Register built-in tasks
@register_task("sudoku")
def create_sudoku_task(config: Any) -> Tuple[Dataset, Dataset]:
    """Create Sudoku train/test datasets."""
    from .data.sudoku import SudokuDataset, SimpleSudokuDataset
    
    train_ds = SudokuDataset(
        n_puzzles=getattr(config, "n_train_samples", 500),
        n_hints_range=(25, 35),
        augment=True,
        n_augmentations=getattr(config, "n_augmentations", 100),
        seed=getattr(config, "seed", 42)
    )
    
    test_ds = SimpleSudokuDataset(
        n_samples=getattr(config, "n_test_samples", 100),
        difficulty="medium"
    )
    
    return train_ds, test_ds
