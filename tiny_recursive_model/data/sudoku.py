"""
Sudoku Dataset
==============

Dataset for training TRM on Sudoku puzzle solving.

Features:
- Generates valid Sudoku puzzles with solutions
- Configurable difficulty (number of hints)
- Data augmentation that preserves Sudoku validity
"""

import torch
import numpy as np
import random
from typing import Tuple, Optional, List
from .base import BaseDataset


def is_valid_sudoku(grid: np.ndarray) -> bool:
    """Check if a 9x9 Sudoku grid has no conflicts."""
    for i in range(9):
        # Check row
        row = grid[i, :]
        row_nonzero = row[row > 0]
        if len(row_nonzero) != len(set(row_nonzero)):
            return False
        
        # Check column
        col = grid[:, i]
        col_nonzero = col[col > 0]
        if len(col_nonzero) != len(set(col_nonzero)):
            return False
    
    # Check 3x3 boxes
    for box_row in range(3):
        for box_col in range(3):
            box = grid[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3]
            box_nonzero = box.flatten()[box.flatten() > 0]
            if len(box_nonzero) != len(set(box_nonzero)):
                return False
    
    return True


def is_complete_sudoku(grid: np.ndarray) -> bool:
    """Check if Sudoku is completely and correctly filled."""
    if 0 in grid:
        return False
    return is_valid_sudoku(grid)


def solve_sudoku(grid: np.ndarray) -> Optional[np.ndarray]:
    """Solve a Sudoku puzzle using backtracking.
    
    Parameters
    ----------
    grid : np.ndarray
        9x9 array with 0 for empty cells
    
    Returns
    -------
    np.ndarray or None
        Solved grid, or None if unsolvable
    """
    grid = grid.copy()
    
    def find_empty():
        for i in range(9):
            for j in range(9):
                if grid[i, j] == 0:
                    return i, j
        return None
    
    def is_valid_move(row, col, num):
        if num in grid[row, :]:
            return False
        if num in grid[:, col]:
            return False
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        if num in grid[box_row:box_row+3, box_col:box_col+3]:
            return False
        return True
    
    def solve():
        pos = find_empty()
        if pos is None:
            return True
        
        row, col = pos
        for num in range(1, 10):
            if is_valid_move(row, col, num):
                grid[row, col] = num
                if solve():
                    return True
                grid[row, col] = 0
        return False
    
    if solve():
        return grid
    return None


def generate_complete_sudoku() -> np.ndarray:
    """Generate a completely filled valid Sudoku grid."""
    grid = np.zeros((9, 9), dtype=np.int32)
    
    # Fill diagonal 3x3 boxes first (they don't affect each other)
    for i in range(0, 9, 3):
        nums = list(range(1, 10))
        random.shuffle(nums)
        for r in range(3):
            for c in range(3):
                grid[i+r, i+c] = nums[r*3 + c]
    
    # Solve the rest
    solved = solve_sudoku(grid)
    if solved is not None:
        return solved
    
    # Fallback: try again
    return generate_complete_sudoku()


def generate_puzzle(solution: np.ndarray, n_hints: int = 25) -> np.ndarray:
    """Create a puzzle from a solution by removing cells.
    
    Parameters
    ----------
    solution : np.ndarray
        Complete 9x9 solution
    n_hints : int
        Number of cells to keep visible
    
    Returns
    -------
    np.ndarray
        Puzzle with 0 for hidden cells
    """
    puzzle = solution.copy()
    positions = [(i, j) for i in range(9) for j in range(9)]
    random.shuffle(positions)
    
    n_to_remove = 81 - n_hints
    for i in range(n_to_remove):
        r, c = positions[i]
        puzzle[r, c] = 0
    
    return puzzle


def augment_sudoku(
    puzzle: np.ndarray, 
    solution: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply validity-preserving augmentation to a Sudoku.
    
    Augmentations applied:
    1. Permute digits (1-9 mapping)
    2. Swap rows within bands
    3. Swap columns within stacks
    4. Swap bands (50% chance)
    5. Swap stacks (50% chance)
    6. Transpose (50% chance)
    
    Parameters
    ----------
    puzzle : np.ndarray
        Input puzzle
    solution : np.ndarray
        Solution grid
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Augmented (puzzle, solution)
    """
    puzzle = puzzle.copy()
    solution = solution.copy()
    
    # 1. Permute digits
    perm = list(range(1, 10))
    random.shuffle(perm)
    perm = [0] + perm  # 0 maps to 0
    
    for grid in [puzzle, solution]:
        for i in range(9):
            for j in range(9):
                grid[i, j] = perm[grid[i, j]]
    
    # 2. Swap rows within bands
    for band in range(3):
        rows = list(range(band*3, band*3 + 3))
        random.shuffle(rows)
        puzzle[band*3:band*3+3, :] = puzzle[rows, :]
        solution[band*3:band*3+3, :] = solution[rows, :]
    
    # 3. Swap columns within stacks
    for stack in range(3):
        cols = list(range(stack*3, stack*3 + 3))
        random.shuffle(cols)
        puzzle[:, stack*3:stack*3+3] = puzzle[:, cols]
        solution[:, stack*3:stack*3+3] = solution[:, cols]
    
    # 4. Swap bands (50% chance)
    if random.random() > 0.5:
        bands = [0, 1, 2]
        random.shuffle(bands)
        new_puzzle = np.zeros_like(puzzle)
        new_solution = np.zeros_like(solution)
        for i, b in enumerate(bands):
            new_puzzle[i*3:(i+1)*3, :] = puzzle[b*3:(b+1)*3, :]
            new_solution[i*3:(i+1)*3, :] = solution[b*3:(b+1)*3, :]
        puzzle, solution = new_puzzle, new_solution
    
    # 5. Swap stacks (50% chance)
    if random.random() > 0.5:
        stacks = [0, 1, 2]
        random.shuffle(stacks)
        new_puzzle = np.zeros_like(puzzle)
        new_solution = np.zeros_like(solution)
        for i, s in enumerate(stacks):
            new_puzzle[:, i*3:(i+1)*3] = puzzle[:, s*3:(s+1)*3]
            new_solution[:, i*3:(i+1)*3] = solution[:, s*3:(s+1)*3]
        puzzle, solution = new_puzzle, new_solution
    
    # 6. Transpose (50% chance)
    if random.random() > 0.5:
        puzzle = puzzle.T.copy()
        solution = solution.T.copy()
    
    return puzzle, solution


def visualize_sudoku(grid: np.ndarray, title: str = "") -> str:
    """Create a string visualization of a Sudoku grid."""
    lines = []
    if title:
        lines.append(title)
        lines.append("=" * 25)
    
    for i in range(9):
        if i > 0 and i % 3 == 0:
            lines.append("------+-------+------")
        
        row_str = ""
        for j in range(9):
            if j > 0 and j % 3 == 0:
                row_str += " | "
            val = grid[i, j]
            row_str += f" {val if val > 0 else '.'}"
        lines.append(row_str)
    
    return "\n".join(lines)


class SudokuDataset(BaseDataset):
    """PyTorch Dataset for Sudoku puzzles with augmentation.
    
    Parameters
    ----------
    n_puzzles : int
        Number of base puzzles to generate
    n_hints_range : Tuple[int, int]
        (min, max) hints per puzzle
    augment : bool
        Whether to apply augmentation
    n_augmentations : int
        Effective dataset size multiplier
    seed : int, optional
        Random seed
    
    Examples
    --------
    >>> dataset = SudokuDataset(n_puzzles=1000, n_augmentations=100)
    >>> print(f"Dataset size: {len(dataset)}")  # 100,000
    >>> puzzle, solution = dataset[0]
    """
    
    VOCAB_SIZE = 10  # 0-9 (0 = empty)
    CONTEXT_LENGTH = 81  # 9x9
    
    def __init__(
        self,
        n_puzzles: int = 1000,
        n_hints_range: Tuple[int, int] = (17, 35),
        augment: bool = True,
        n_augmentations: int = 100,
        seed: Optional[int] = None
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self._augment = augment
        self.n_augmentations = n_augmentations if augment else 1
        
        # Generate base puzzles
        self.puzzles = []
        self.solutions = []
        
        for _ in range(n_puzzles):
            solution = generate_complete_sudoku()
            n_hints = random.randint(n_hints_range[0], n_hints_range[1])
            puzzle = generate_puzzle(solution, n_hints)
            self.puzzles.append(puzzle)
            self.solutions.append(solution)
    
    def __len__(self) -> int:
        return len(self.puzzles) * self.n_augmentations
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        base_idx = idx // self.n_augmentations
        puzzle = self.puzzles[base_idx]
        solution = self.solutions[base_idx]
        
        if self._augment:
            puzzle, solution = augment_sudoku(puzzle, solution)
        
        puzzle_flat = torch.tensor(puzzle.flatten(), dtype=torch.long)
        solution_flat = torch.tensor(solution.flatten(), dtype=torch.long)
        
        return puzzle_flat, solution_flat
    
    @staticmethod
    def get_vocab_size() -> int:
        return SudokuDataset.VOCAB_SIZE
    
    @staticmethod
    def get_context_length() -> int:
        return SudokuDataset.CONTEXT_LENGTH
    
    def visualize(self, idx: int) -> str:
        """Visualize a puzzle and its solution."""
        base_idx = idx // self.n_augmentations
        puzzle = self.puzzles[base_idx]
        solution = self.solutions[base_idx]
        
        return (
            visualize_sudoku(puzzle, "Puzzle") + 
            "\n\n" + 
            visualize_sudoku(solution, "Solution")
        )


class SimpleSudokuDataset(BaseDataset):
    """Simplified Sudoku dataset without augmentation.
    
    Useful for quick testing and evaluation.
    
    Parameters
    ----------
    n_samples : int
        Number of puzzles
    difficulty : str
        'easy', 'medium', or 'hard'
    """
    
    def __init__(self, n_samples: int = 100, difficulty: str = "medium"):
        hint_ranges = {
            "easy": (30, 35),
            "medium": (25, 30),
            "hard": (17, 24)
        }
        n_hints_range = hint_ranges.get(difficulty, (25, 30))
        
        self.data = []
        for _ in range(n_samples):
            solution = generate_complete_sudoku()
            n_hints = random.randint(*n_hints_range)
            puzzle = generate_puzzle(solution, n_hints)
            self.data.append((
                torch.tensor(puzzle.flatten(), dtype=torch.long),
                torch.tensor(solution.flatten(), dtype=torch.long)
            ))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @staticmethod
    def get_vocab_size() -> int:
        return 10
    
    @staticmethod
    def get_context_length() -> int:
        return 81
