"""
Data Module
===========

Base dataset classes and implementations for various tasks.

Tasks from TRM paper:
- Sudoku-Extreme: 87.4% accuracy (MLP-Mixer, fixed 9x9 grid)
- Maze-Hard: 85.3% accuracy (Attention, 30x30 grid)
- ARC-AGI-1: 44.6% accuracy (Attention, variable grids)
- ARC-AGI-2: 7.8% accuracy (harder version)

Additional tasks:
- Arithmetic: 3-digit addition (simpler, for testing)
"""

from .base import BaseDataset
from .sudoku import SudokuDataset, SimpleSudokuDataset
from .arithmetic import ArithmeticDataset, MultiplicationDataset
from .maze import MazeHardDataset, MazeSimpleDataset
from .arc import ARCDataset, ARCFewShotDataset

__all__ = [
    "BaseDataset",
    # Sudoku
    "SudokuDataset",
    "SimpleSudokuDataset",
    # Maze
    "MazeHardDataset",
    "MazeSimpleDataset",
    # ARC
    "ARCDataset",
    "ARCFewShotDataset",
    # Arithmetic
    "ArithmeticDataset",
    "MultiplicationDataset",
]
