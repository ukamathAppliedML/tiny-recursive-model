"""
ARC-AGI Dataset for TRM
=======================

Abstraction and Reasoning Corpus (ARC) dataset implementation.
This is the benchmark where TRM achieved 44.6% on ARC-AGI-1.

ARC tasks are visual reasoning puzzles:
- Given input-output grid pairs as examples
- Predict the output for a new input
- Each task has a different underlying rule

Dataset: https://github.com/fchollet/ARC-AGI

The challenge:
- Tasks are diverse (each has different transformation rules)
- Few-shot learning (2-4 example pairs per task)
- Variable grid sizes (up to 30x30)
- 10 colors (0-9)

For TRM, we format as:
- Flatten grids to sequences
- Pad to fixed size (30x30 = 900 tokens)
- Model learns to predict output from input

Usage:
    python -m tiny_recursive_model.data.arc  # Download and test
    
    # Or in code:
    from tiny_recursive_model.data.arc import ARCDataset
    dataset = ARCDataset(split='training')
"""

import json
import os
import urllib.request
import zipfile
import torch
import numpy as np
import random
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from .base import BaseDataset


# ARC dataset URLs
ARC_REPO_URL = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"
ARC_DATA_DIR = Path.home() / ".cache" / "arc-agi"


def download_arc_dataset(force: bool = False) -> Path:
    """Download ARC-AGI dataset if not present.
    
    Returns path to data directory.
    """
    data_dir = ARC_DATA_DIR / "ARC-AGI-master" / "data"
    
    if data_dir.exists() and not force:
        return data_dir
    
    print("Downloading ARC-AGI dataset...")
    ARC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    zip_path = ARC_DATA_DIR / "arc.zip"
    
    # Download
    urllib.request.urlretrieve(ARC_REPO_URL, zip_path)
    
    # Extract
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(ARC_DATA_DIR)
    
    # Cleanup
    zip_path.unlink()
    
    print(f"Dataset downloaded to {data_dir}")
    return data_dir


def load_arc_tasks(split: str = "training", data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load ARC tasks from JSON files.
    
    Parameters
    ----------
    split : str
        'training' or 'evaluation'
    data_dir : Path, optional
        Path to ARC data directory
    
    Returns
    -------
    Dict[str, Any]
        Dictionary mapping task_id -> task data
    """
    if data_dir is None:
        data_dir = download_arc_dataset()
    
    split_dir = data_dir / split
    
    tasks = {}
    for json_file in sorted(split_dir.glob("*.json")):
        task_id = json_file.stem
        with open(json_file) as f:
            tasks[task_id] = json.load(f)
    
    return tasks


def grid_to_flat(grid: List[List[int]], max_size: int = 30) -> np.ndarray:
    """Convert 2D grid to padded flat array.
    
    Padding token is 10 (colors are 0-9).
    """
    grid = np.array(grid)
    h, w = grid.shape
    
    # Create padded grid
    padded = np.full((max_size, max_size), 10, dtype=np.int64)  # 10 = PAD
    padded[:h, :w] = grid
    
    return padded.flatten()


def flat_to_grid(flat: np.ndarray, original_shape: Tuple[int, int], max_size: int = 30) -> np.ndarray:
    """Convert flat array back to 2D grid."""
    grid = flat.reshape(max_size, max_size)
    h, w = original_shape
    return grid[:h, :w]


class ARCDataset(BaseDataset):
    """ARC-AGI Dataset for TRM.
    
    Each sample is an (input_grid, output_grid) pair from ARC tasks.
    
    Parameters
    ----------
    split : str
        'training' or 'evaluation'
    max_grid_size : int
        Maximum grid dimension (grids are padded to this size)
    augment : bool
        Whether to apply augmentation (rotations, flips, color permutations)
    n_augmentations : int
        Multiplier for dataset size via augmentation
    include_examples : bool
        If True, include training examples from each task.
        If False, only include test pairs.
    seed : int, optional
        Random seed
    
    Examples
    --------
    >>> dataset = ARCDataset(split='training')
    >>> print(f"Number of samples: {len(dataset)}")
    >>> x, y = dataset[0]
    >>> print(f"Input shape: {x.shape}")  # [900]
    """
    
    # Vocabulary: 0-9 colors + PAD (10)
    VOCAB_SIZE = 11
    
    def __init__(
        self,
        split: str = "training",
        max_grid_size: int = 30,
        augment: bool = True,
        n_augmentations: int = 8,
        include_examples: bool = True,
        data_dir: Optional[Path] = None,
        seed: Optional[int] = None
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.max_grid_size = max_grid_size
        self.context_length = max_grid_size * max_grid_size
        self.augment = augment
        self.n_augmentations = n_augmentations if augment else 1
        
        # Load tasks
        self.tasks = load_arc_tasks(split, data_dir)
        
        # Extract all (input, output) pairs
        self.samples = []
        self.sample_metadata = []  # Store original shapes for visualization
        
        for task_id, task_data in self.tasks.items():
            # Training examples
            if include_examples:
                for i, example in enumerate(task_data.get('train', [])):
                    inp = example['input']
                    out = example['output']
                    self.samples.append((inp, out))
                    self.sample_metadata.append({
                        'task_id': task_id,
                        'type': 'train',
                        'index': i,
                        'input_shape': (len(inp), len(inp[0])),
                        'output_shape': (len(out), len(out[0]))
                    })
            
            # Test examples
            for i, example in enumerate(task_data.get('test', [])):
                inp = example['input']
                out = example['output']
                self.samples.append((inp, out))
                self.sample_metadata.append({
                    'task_id': task_id,
                    'type': 'test',
                    'index': i,
                    'input_shape': (len(inp), len(inp[0])),
                    'output_shape': (len(out), len(out[0]))
                })
        
        print(f"Loaded {len(self.samples)} samples from {len(self.tasks)} tasks ({split})")
    
    def __len__(self) -> int:
        return len(self.samples) * self.n_augmentations
    
    def _augment_grid(self, grid: np.ndarray) -> np.ndarray:
        """Apply random augmentation to a grid."""
        # Random rotation (0, 90, 180, 270 degrees)
        k = random.randint(0, 3)
        grid = np.rot90(grid, k)
        
        # Random flip
        if random.random() > 0.5:
            grid = np.fliplr(grid)
        if random.random() > 0.5:
            grid = np.flipud(grid)
        
        return grid
    
    def _augment_colors(self, inp: np.ndarray, out: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply consistent color permutation to both grids."""
        # Create a random permutation for colors 0-9
        perm = list(range(10))
        random.shuffle(perm)
        
        # Apply to both grids
        inp_aug = inp.copy()
        out_aug = out.copy()
        
        for old_color, new_color in enumerate(perm):
            inp_aug[inp == old_color] = new_color
            out_aug[out == old_color] = new_color
        
        return inp_aug, out_aug
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        base_idx = idx // self.n_augmentations
        inp_grid, out_grid = self.samples[base_idx]
        
        # Convert to numpy
        inp = np.array(inp_grid, dtype=np.int64)
        out = np.array(out_grid, dtype=np.int64)
        
        # Apply augmentation
        if self.augment and idx % self.n_augmentations > 0:
            # Spatial augmentation (must be same for both)
            k = random.randint(0, 3)
            inp = np.rot90(inp, k)
            out = np.rot90(out, k)
            
            if random.random() > 0.5:
                inp = np.fliplr(inp)
                out = np.fliplr(out)
            if random.random() > 0.5:
                inp = np.flipud(inp)
                out = np.flipud(out)
            
            # Color permutation
            inp, out = self._augment_colors(inp, out)
        
        # Pad and flatten
        inp_flat = grid_to_flat(inp, self.max_grid_size)
        out_flat = grid_to_flat(out, self.max_grid_size)
        
        return (
            torch.tensor(inp_flat, dtype=torch.long),
            torch.tensor(out_flat, dtype=torch.long)
        )
    
    @staticmethod
    def get_vocab_size() -> int:
        return ARCDataset.VOCAB_SIZE
    
    def get_context_length(self) -> int:
        return self.context_length
    
    def visualize(self, idx: int) -> str:
        """Visualize a sample as colored grid."""
        base_idx = idx // self.n_augmentations
        inp_grid, out_grid = self.samples[base_idx]
        meta = self.sample_metadata[base_idx]
        
        # Color map for terminal
        colors = ['⬛', '🟦', '🟥', '🟩', '🟨', '⬜', '🟪', '🟧', '🩵', '🟫']
        
        def grid_to_str(grid):
            return '\n'.join(''.join(colors[c] if c < 10 else '·' for c in row) for row in grid)
        
        return (
            f"Task: {meta['task_id']} ({meta['type']} #{meta['index']})\n"
            f"Input ({meta['input_shape'][0]}x{meta['input_shape'][1]}):\n"
            f"{grid_to_str(inp_grid)}\n"
            f"Output ({meta['output_shape'][0]}x{meta['output_shape'][1]}):\n"
            f"{grid_to_str(out_grid)}"
        )
    
    def get_task_samples(self, task_id: str) -> List[int]:
        """Get all sample indices for a specific task."""
        indices = []
        for i, meta in enumerate(self.sample_metadata):
            if meta['task_id'] == task_id:
                indices.append(i * self.n_augmentations)
        return indices


class ARCFewShotDataset(BaseDataset):
    """ARC dataset formatted for few-shot learning.
    
    Each sample includes the training examples as context,
    followed by the test input, and expects the test output.
    
    This is closer to how ARC is actually evaluated.
    """
    
    VOCAB_SIZE = 12  # 0-9 colors, PAD (10), SEP (11)
    
    def __init__(
        self,
        split: str = "training",
        max_grid_size: int = 10,  # Smaller grids for context length
        max_examples: int = 3,
        seed: Optional[int] = None
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.max_grid_size = max_grid_size
        self.max_examples = max_examples
        
        # Each sample: [ex1_in, SEP, ex1_out, SEP, ex2_in, SEP, ex2_out, SEP, test_in, SEP, PAD...]
        # With max 3 examples and grid size 10x10:
        # 3 * (100 + 1 + 100 + 1) + 100 + 1 + 100 = 809 tokens
        tokens_per_pair = max_grid_size * max_grid_size + 1 + max_grid_size * max_grid_size + 1
        self.context_length = max_examples * tokens_per_pair + max_grid_size * max_grid_size + 1 + max_grid_size * max_grid_size
        
        self.tasks = load_arc_tasks(split)
        
        # Filter tasks with small enough grids
        self.valid_tasks = []
        for task_id, task_data in self.tasks.items():
            valid = True
            for example in task_data.get('train', []) + task_data.get('test', []):
                inp = example['input']
                out = example['output']
                if len(inp) > max_grid_size or len(inp[0]) > max_grid_size:
                    valid = False
                if len(out) > max_grid_size or len(out[0]) > max_grid_size:
                    valid = False
            if valid and len(task_data.get('train', [])) >= 1:
                self.valid_tasks.append(task_id)
        
        print(f"Found {len(self.valid_tasks)} tasks with grids <= {max_grid_size}x{max_grid_size}")
    
    def __len__(self) -> int:
        return len(self.valid_tasks)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        task_id = self.valid_tasks[idx]
        task = self.tasks[task_id]
        
        SEP = 11
        PAD = 10
        
        # Get examples
        train_examples = task['train'][:self.max_examples]
        test_example = task['test'][0]  # Use first test
        
        input_tokens = []
        target_tokens = []
        
        # Add training examples
        for ex in train_examples:
            inp_flat = grid_to_flat(ex['input'], self.max_grid_size)
            out_flat = grid_to_flat(ex['output'], self.max_grid_size)
            
            input_tokens.extend(inp_flat.tolist())
            input_tokens.append(SEP)
            input_tokens.extend(out_flat.tolist())
            input_tokens.append(SEP)
            
            target_tokens.extend(inp_flat.tolist())
            target_tokens.append(SEP)
            target_tokens.extend(out_flat.tolist())
            target_tokens.append(SEP)
        
        # Add test input
        test_inp_flat = grid_to_flat(test_example['input'], self.max_grid_size)
        test_out_flat = grid_to_flat(test_example['output'], self.max_grid_size)
        
        input_tokens.extend(test_inp_flat.tolist())
        input_tokens.append(SEP)
        input_tokens.extend([PAD] * len(test_out_flat))  # Mask output in input
        
        target_tokens.extend(test_inp_flat.tolist())
        target_tokens.append(SEP)
        target_tokens.extend(test_out_flat.tolist())  # Actual output in target
        
        # Pad to context length
        while len(input_tokens) < self.context_length:
            input_tokens.append(PAD)
            target_tokens.append(PAD)
        
        # Truncate if needed
        input_tokens = input_tokens[:self.context_length]
        target_tokens = target_tokens[:self.context_length]
        
        return (
            torch.tensor(input_tokens, dtype=torch.long),
            torch.tensor(target_tokens, dtype=torch.long)
        )
    
    @staticmethod
    def get_vocab_size() -> int:
        return ARCFewShotDataset.VOCAB_SIZE
    
    def get_context_length(self) -> int:
        return self.context_length


if __name__ == "__main__":
    print("=" * 60)
    print("ARC-AGI Dataset Test")
    print("=" * 60)
    
    # Download and load
    print("\nLoading ARC training set...")
    dataset = ARCDataset(split='training', augment=False)
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Vocab size: {dataset.get_vocab_size()}")
    print(f"Context length: {dataset.get_context_length()}")
    print(f"Number of tasks: {len(dataset.tasks)}")
    
    # Show a sample
    print("\n" + "=" * 60)
    print("Sample visualization:")
    print("=" * 60)
    print(dataset.visualize(0))
    
    # Test tensor shapes
    x, y = dataset[0]
    print(f"\nTensor shapes:")
    print(f"  Input: {x.shape}, dtype: {x.dtype}")
    print(f"  Target: {y.shape}, dtype: {y.dtype}")
    
    # Test few-shot dataset
    print("\n" + "=" * 60)
    print("Few-shot Dataset Test")
    print("=" * 60)
    
    fewshot = ARCFewShotDataset(split='training', max_grid_size=10)
    print(f"Tasks with small grids: {len(fewshot)}")
    print(f"Context length: {fewshot.get_context_length()}")
    
    if len(fewshot) > 0:
        x, y = fewshot[0]
        print(f"Sample shape: {x.shape}")
    
    print("\n✓ ARC dataset ready!")
