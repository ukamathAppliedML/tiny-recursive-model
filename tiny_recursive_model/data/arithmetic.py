"""
Arithmetic Task for TRM
=======================

A simpler task than Sudoku that trains faster and demonstrates
TRM's recursive reasoning on learning addition.

Task: Given two numbers encoded as digit sequences, predict their sum.
Example: [3,4,2] + [1,5,8] = [5,0,0] (342 + 158 = 500)

This task requires "carrying" which benefits from iterative refinement -
the model needs to propagate carry information across digits.

Usage:
    python -m tiny_recursive_model.data.arithmetic  # Test the dataset
    
To train (add to __main__.py or run directly):
    from tiny_recursive_model.data.arithmetic import ArithmeticDataset
    # ... create dataset and train
"""

import torch
import numpy as np
import random
from typing import Tuple, Optional, List
from .base import BaseDataset


class ArithmeticDataset(BaseDataset):
    """Dataset for learning multi-digit addition.
    
    Input format: [d1, d2, d3, SEP, d4, d5, d6, EQ, PAD, PAD, PAD]
    Target format: [d1, d2, d3, SEP, d4, d5, d6, EQ, r1, r2, r3]
    
    Where:
    - d1-d3: First number (e.g., 342)
    - SEP (10): Separator token (+)
    - d4-d6: Second number (e.g., 158)  
    - EQ (11): Equals token (=)
    - r1-r3: Result (e.g., 500)
    - PAD (12): Padding in input, replaced by result in target
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    n_digits : int
        Number of digits per operand (default: 3)
    augment : bool
        Whether to apply augmentation (operand swapping)
    seed : int, optional
        Random seed
    
    Examples
    --------
    >>> dataset = ArithmeticDataset(n_samples=1000, n_digits=3)
    >>> x, y = dataset[0]
    >>> print(f"Input: {x.tolist()}")
    >>> print(f"Target: {y.tolist()}")
    """
    
    # Vocabulary: 0-9 digits, +, =, PAD
    DIGIT_TOKENS = list(range(10))  # 0-9
    SEP_TOKEN = 10   # + 
    EQ_TOKEN = 11    # =
    PAD_TOKEN = 12   # padding
    
    VOCAB_SIZE = 13
    
    def __init__(
        self,
        n_samples: int = 1000,
        n_digits: int = 3,
        augment: bool = True,
        seed: Optional[int] = None
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.n_digits = n_digits
        self.augment = augment
        self.context_length = n_digits + 1 + n_digits + 1 + n_digits  # num1 + sep + num2 + eq + result
        
        self.samples = []
        for _ in range(n_samples):
            self.samples.append(self._generate_sample())
    
    def _generate_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single addition problem."""
        max_val = 10 ** self.n_digits - 1
        # Ensure sum doesn't overflow (stays within n_digits)
        max_operand = max_val // 2
        
        num1 = random.randint(0, max_operand)
        num2 = random.randint(0, max_operand)
        result = num1 + num2
        
        # Convert to digit arrays (padded with leading zeros)
        def to_digits(n: int) -> List[int]:
            s = str(n).zfill(self.n_digits)
            return [int(d) for d in s]
        
        digits1 = to_digits(num1)
        digits2 = to_digits(num2)
        digits_result = to_digits(result)
        
        # Build sequences
        # Input: [num1..., SEP, num2..., EQ, PAD...]
        # Target: [num1..., SEP, num2..., EQ, result...]
        
        input_seq = digits1 + [self.SEP_TOKEN] + digits2 + [self.EQ_TOKEN] + [self.PAD_TOKEN] * self.n_digits
        target_seq = digits1 + [self.SEP_TOKEN] + digits2 + [self.EQ_TOKEN] + digits_result
        
        return np.array(input_seq, dtype=np.int64), np.array(target_seq, dtype=np.int64)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq, target_seq = self.samples[idx]
        
        # Augmentation: swap operands (a + b = b + a)
        if self.augment and random.random() > 0.5:
            # Swap num1 and num2
            n = self.n_digits
            input_seq = input_seq.copy()
            target_seq = target_seq.copy()
            
            # Swap: [0:n] <-> [n+1:2n+1]
            temp = input_seq[:n].copy()
            input_seq[:n] = input_seq[n+1:2*n+1]
            input_seq[n+1:2*n+1] = temp
            
            temp = target_seq[:n].copy()
            target_seq[:n] = target_seq[n+1:2*n+1]
            target_seq[n+1:2*n+1] = temp
        
        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long)
        )
    
    @staticmethod
    def get_vocab_size() -> int:
        return ArithmeticDataset.VOCAB_SIZE
    
    def get_context_length(self) -> int:
        return self.context_length
    
    def visualize(self, idx: int) -> str:
        """Pretty print a sample."""
        input_seq, target_seq = self.samples[idx]
        n = self.n_digits
        
        num1 = ''.join(str(d) for d in input_seq[:n])
        num2 = ''.join(str(d) for d in input_seq[n+1:2*n+1])
        result = ''.join(str(d) for d in target_seq[2*n+2:])
        
        return f"{num1} + {num2} = {result}"


class MultiplicationDataset(BaseDataset):
    """Dataset for learning single-digit multiplication table.
    
    Simpler than addition - good for quick tests.
    
    Input: [a, ×, b, =, PAD]
    Target: [a, ×, b, =, result (2 digits)]
    """
    
    VOCAB_SIZE = 13  # 0-9, ×(10), =(11), PAD(12)
    CONTEXT_LENGTH = 6  # a, ×, b, =, r1, r2
    
    def __init__(self, n_samples: int = 1000, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.samples = []
        for _ in range(n_samples):
            a = random.randint(0, 9)
            b = random.randint(0, 9)
            result = a * b
            
            r1 = result // 10
            r2 = result % 10
            
            input_seq = [a, 10, b, 11, 12, 12]
            target_seq = [a, 10, b, 11, r1, r2]
            
            self.samples.append((
                np.array(input_seq, dtype=np.int64),
                np.array(target_seq, dtype=np.int64)
            ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        inp, tgt = self.samples[idx]
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)
    
    @staticmethod
    def get_vocab_size():
        return MultiplicationDataset.VOCAB_SIZE
    
    @staticmethod
    def get_context_length():
        return MultiplicationDataset.CONTEXT_LENGTH
    
    def visualize(self, idx: int) -> str:
        inp, tgt = self.samples[idx]
        a, b = inp[0], inp[2]
        result = tgt[4] * 10 + tgt[5]
        return f"{a} × {b} = {result}"


if __name__ == "__main__":
    print("=" * 50)
    print("Arithmetic Dataset Test")
    print("=" * 50)
    
    # Test ArithmeticDataset
    print("\n3-digit Addition:")
    dataset = ArithmeticDataset(n_samples=5, n_digits=3, augment=False, seed=42)
    print(f"  Vocab size: {dataset.get_vocab_size()}")
    print(f"  Context length: {dataset.get_context_length()}")
    print(f"  Samples:")
    for i in range(5):
        print(f"    {dataset.visualize(i)}")
        x, y = dataset[i]
        print(f"      Input:  {x.tolist()}")
        print(f"      Target: {y.tolist()}")
    
    # Test MultiplicationDataset
    print("\n\nMultiplication Table:")
    mult_dataset = MultiplicationDataset(n_samples=5, seed=42)
    print(f"  Vocab size: {mult_dataset.get_vocab_size()}")
    print(f"  Context length: {mult_dataset.get_context_length()}")
    print(f"  Samples:")
    for i in range(5):
        print(f"    {mult_dataset.visualize(i)}")
    
    print("\n✓ Arithmetic datasets working!")
