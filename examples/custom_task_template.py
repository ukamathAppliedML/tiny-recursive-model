"""
Custom Task Template
====================

Template for creating new tasks for TRM.

To add a new task:
1. Copy this file and rename it
2. Implement the dataset class
3. Register the task
4. (Optional) Add a model config preset

Example usage after implementation:
    trm train --task my_custom_task --epochs 50
"""

import torch
import numpy as np
import random
from typing import Tuple, Optional, List
from tiny_recursive_model.data.base import BaseDataset
from tiny_recursive_model.tasks import register_task
from tiny_recursive_model.config import TRMConfig


class CustomTaskDataset(BaseDataset):
    """Template dataset for a custom task.
    
    Replace this with your actual task implementation.
    
    Key requirements:
    1. Input and output should be fixed-length token sequences
    2. Tokens should be integers from 0 to vocab_size-1
    3. The task should benefit from iterative refinement
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    difficulty : str
        Task difficulty level
    augment : bool
        Whether to apply data augmentation
    seed : int, optional
        Random seed for reproducibility
    """
    
    # Define vocabulary
    VOCAB_SIZE = 10  # Number of unique tokens
    CONTEXT_LENGTH = 100  # Sequence length
    
    def __init__(
        self,
        n_samples: int = 1000,
        difficulty: str = "medium",
        augment: bool = True,
        seed: Optional[int] = None
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.augment = augment
        self.difficulty = difficulty
        
        # Generate samples
        self.samples = []
        for _ in range(n_samples):
            input_seq, target_seq = self._generate_sample()
            self.samples.append((input_seq, target_seq))
    
    def _generate_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single (input, target) pair.
        
        Override this method with your task's data generation logic.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - input_seq: Input token sequence
            - target_seq: Target token sequence
        """
        # REPLACE THIS with your actual generation logic
        # This is just a placeholder that generates random data
        
        # Example: Copy task with noise
        target = np.random.randint(1, self.VOCAB_SIZE, self.CONTEXT_LENGTH)
        
        # Add some "noise" to create input (mask some tokens)
        input_seq = target.copy()
        n_mask = int(0.3 * self.CONTEXT_LENGTH)  # 30% masked
        mask_indices = np.random.choice(
            self.CONTEXT_LENGTH, n_mask, replace=False
        )
        input_seq[mask_indices] = 0  # 0 = mask token
        
        return input_seq, target
    
    def _augment(
        self, 
        input_seq: np.ndarray, 
        target_seq: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation.
        
        Override this method with task-specific augmentation.
        Good augmentations preserve the input-output relationship
        while increasing diversity.
        
        Parameters
        ----------
        input_seq : np.ndarray
            Input sequence
        target_seq : np.ndarray
            Target sequence
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Augmented (input, target) pair
        """
        # REPLACE THIS with your actual augmentation logic
        
        # Example: Random token permutation
        if random.random() > 0.5:
            # Create a permutation for non-zero tokens
            perm = list(range(1, self.VOCAB_SIZE))
            random.shuffle(perm)
            perm = [0] + perm  # 0 maps to 0
            
            input_seq = np.array([perm[x] for x in input_seq])
            target_seq = np.array([perm[x] for x in target_seq])
        
        return input_seq, target_seq
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq, target_seq = self.samples[idx]
        
        if self.augment:
            input_seq, target_seq = self._augment(
                input_seq.copy(), target_seq.copy()
            )
        
        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long)
        )
    
    @staticmethod
    def get_vocab_size() -> int:
        return CustomTaskDataset.VOCAB_SIZE
    
    @staticmethod
    def get_context_length() -> int:
        return CustomTaskDataset.CONTEXT_LENGTH
    
    def visualize(self, idx: int) -> str:
        """Visualize a sample."""
        input_seq, target_seq = self.samples[idx]
        return (
            f"Input:  {input_seq[:20]}...\n"
            f"Target: {target_seq[:20]}..."
        )


# Register the task
@register_task("custom")
def create_custom_task(config):
    """Factory function for custom task datasets."""
    train_ds = CustomTaskDataset(
        n_samples=getattr(config, "n_train_samples", 500),
        difficulty="medium",
        augment=True,
        seed=getattr(config, "seed", 42)
    )
    
    test_ds = CustomTaskDataset(
        n_samples=getattr(config, "n_test_samples", 100),
        difficulty="medium",
        augment=False
    )
    
    return train_ds, test_ds


# Define model config preset (optional but recommended)
CUSTOM_TASK_CONFIG = TRMConfig(
    vocab_size=CustomTaskDataset.VOCAB_SIZE,
    hidden_dim=256,
    num_layers=2,
    context_length=CustomTaskDataset.CONTEXT_LENGTH,
    n_recursions=6,
    T_steps=3,
    n_supervision=16,
    use_attention=False  # Use True for larger context lengths
)


if __name__ == "__main__":
    """Quick test of the custom dataset."""
    
    print("Testing CustomTaskDataset...")
    
    # Create dataset
    dataset = CustomTaskDataset(n_samples=10)
    print(f"Dataset size: {len(dataset)}")
    print(f"Vocab size: {dataset.get_vocab_size()}")
    print(f"Context length: {dataset.get_context_length()}")
    
    # Get a sample
    input_seq, target_seq = dataset[0]
    print(f"Input shape: {input_seq.shape}")
    print(f"Target shape: {target_seq.shape}")
    
    # Visualize
    print("\nSample visualization:")
    print(dataset.visualize(0))
    
    print("\n✓ Custom task template is working!")
