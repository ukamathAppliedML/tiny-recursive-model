"""
Base Dataset Class
==================

Abstract base class for TRM datasets with common utilities.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """Abstract base class for TRM datasets.
    
    Extend this class to create datasets for new tasks.
    
    Required Methods
    ----------------
    __len__ : Return dataset size
    __getitem__ : Return (input_tokens, target_tokens) tuple
    get_vocab_size : Return vocabulary size
    get_context_length : Return sequence length
    
    Optional Methods
    ----------------
    augment : Apply data augmentation
    visualize : Pretty-print a sample
    
    Examples
    --------
    >>> class MyDataset(BaseDataset):
    ...     def __init__(self, n_samples):
    ...         self.samples = [self._generate() for _ in range(n_samples)]
    ...     
    ...     def __len__(self):
    ...         return len(self.samples)
    ...     
    ...     def __getitem__(self, idx):
    ...         return self.samples[idx]
    ...     
    ...     @staticmethod
    ...     def get_vocab_size():
    ...         return 10
    ...     
    ...     @staticmethod
    ...     def get_context_length():
    ...         return 81
    """
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (input_tokens, target_tokens) for a sample.
        
        Parameters
        ----------
        idx : int
            Sample index
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - input_tokens: [context_length] long tensor
            - target_tokens: [context_length] long tensor
        """
        pass
    
    @staticmethod
    @abstractmethod
    def get_vocab_size() -> int:
        """Return the vocabulary size for this task.
        
        This determines the model's vocab_size config.
        """
        pass
    
    @staticmethod
    @abstractmethod
    def get_context_length() -> int:
        """Return the sequence length for this task.
        
        This determines the model's context_length config.
        """
        pass
    
    def augment(
        self, 
        input_tokens: torch.Tensor, 
        target_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation to a sample.
        
        Override this method to implement task-specific augmentation.
        Default implementation returns unchanged inputs.
        
        Parameters
        ----------
        input_tokens : torch.Tensor
            Input sequence
        target_tokens : torch.Tensor
            Target sequence
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Augmented (input, target) pair
        """
        return input_tokens, target_tokens
    
    def visualize(self, sample: Any) -> str:
        """Create a string visualization of a sample.
        
        Override for task-specific visualization.
        
        Parameters
        ----------
        sample : Any
            A sample from the dataset
        
        Returns
        -------
        str
            Human-readable representation
        """
        return str(sample)
    
    def validate(self, input_tokens: torch.Tensor, target_tokens: torch.Tensor) -> bool:
        """Validate that a sample is correct.
        
        Override for task-specific validation.
        
        Parameters
        ----------
        input_tokens : torch.Tensor
            Input sequence
        target_tokens : torch.Tensor
            Target sequence
        
        Returns
        -------
        bool
            True if the sample is valid
        """
        return True
