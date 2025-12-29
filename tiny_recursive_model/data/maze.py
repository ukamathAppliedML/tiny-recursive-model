"""
Maze-Hard Dataset for TRM
=========================

30x30 maze pathfinding task from the TRM paper.
TRM achieves 85.3% accuracy on this benchmark.

Task: Given a maze with start (S) and end (E), predict the path.

Grid encoding:
- 0: Empty cell
- 1: Wall
- 2: Start (S)
- 3: End (E)
- 4: Path (in solution)

The model learns to find the shortest path through the maze.
This requires recursive reasoning to propagate path information.

Usage:
    python -m tiny_recursive_model.data.maze  # Generate and test
    
Training (paper settings):
    python train_maze.py --epochs 50000 --n-train 1000 --n-augmentations 8
"""

import torch
import numpy as np
import random
from typing import Tuple, Optional, List, Set
from collections import deque
from .base import BaseDataset


class MazeHardDataset(BaseDataset):
    """Maze-Hard dataset for TRM.
    
    Generates random mazes and their solutions (shortest paths).
    
    Parameters
    ----------
    n_samples : int
        Number of base mazes to generate
    grid_size : int
        Size of maze grid (default: 30 for Maze-Hard)
    wall_density : float
        Probability of a cell being a wall (default: 0.3)
    min_path_length : int
        Minimum path length for "hard" mazes (default: 20)
    n_augmentations : int
        Number of augmentations per maze (rotations, flips)
    seed : int, optional
        Random seed
    
    Examples
    --------
    >>> dataset = MazeHardDataset(n_samples=1000, grid_size=30)
    >>> x, y = dataset[0]
    >>> print(f"Input shape: {x.shape}")  # [900]
    """
    
    # Vocabulary
    EMPTY = 0
    WALL = 1
    START = 2
    END = 3
    PATH = 4
    
    VOCAB_SIZE = 5
    
    def __init__(
        self,
        n_samples: int = 1000,
        grid_size: int = 30,
        wall_density: float = 0.3,
        min_path_length: int = 20,
        n_augmentations: int = 8,
        seed: Optional[int] = None
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.grid_size = grid_size
        self.context_length = grid_size * grid_size
        self.wall_density = wall_density
        self.min_path_length = min_path_length
        self.n_augmentations = n_augmentations
        
        # Generate mazes
        self.mazes = []
        self.solutions = []
        
        print(f"Generating {n_samples} mazes...")
        attempts = 0
        max_attempts = n_samples * 10
        
        while len(self.mazes) < n_samples and attempts < max_attempts:
            attempts += 1
            maze, solution, path_length = self._generate_maze()
            
            if solution is not None and path_length >= min_path_length:
                self.mazes.append(maze)
                self.solutions.append(solution)
                
                if len(self.mazes) % 100 == 0:
                    print(f"  Generated {len(self.mazes)}/{n_samples} mazes")
        
        print(f"Generated {len(self.mazes)} valid mazes")
    
    def _generate_maze(self) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
        """Generate a random maze with solution.
        
        Returns (maze, solution, path_length) where solution is None if unsolvable.
        """
        n = self.grid_size
        
        # Create maze with random walls
        maze = np.zeros((n, n), dtype=np.int64)
        
        # Add random walls
        wall_mask = np.random.random((n, n)) < self.wall_density
        maze[wall_mask] = self.WALL
        
        # Clear borders for start/end placement
        maze[0, :] = self.EMPTY
        maze[-1, :] = self.EMPTY
        maze[:, 0] = self.EMPTY
        maze[:, -1] = self.EMPTY
        
        # Random start and end on opposite sides
        side = random.randint(0, 3)
        
        if side == 0:  # Start top, end bottom
            start = (0, random.randint(0, n-1))
            end = (n-1, random.randint(0, n-1))
        elif side == 1:  # Start bottom, end top
            start = (n-1, random.randint(0, n-1))
            end = (0, random.randint(0, n-1))
        elif side == 2:  # Start left, end right
            start = (random.randint(0, n-1), 0)
            end = (random.randint(0, n-1), n-1)
        else:  # Start right, end left
            start = (random.randint(0, n-1), n-1)
            end = (random.randint(0, n-1), 0)
        
        # Ensure start and end are not walls
        maze[start] = self.START
        maze[end] = self.END
        
        # Find shortest path using BFS
        path = self._find_path(maze, start, end)
        
        if path is None:
            return maze, None, 0
        
        # Create solution with path marked
        solution = maze.copy()
        for r, c in path:
            if solution[r, c] == self.EMPTY:
                solution[r, c] = self.PATH
        
        return maze, solution, len(path)
    
    def _find_path(
        self, 
        maze: np.ndarray, 
        start: Tuple[int, int], 
        end: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """Find shortest path using BFS."""
        n = self.grid_size
        
        # BFS
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            (r, c), path = queue.popleft()
            
            if (r, c) == end:
                return path
            
            # Check neighbors (4-connected)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < n and 0 <= nc < n:
                    if (nr, nc) not in visited and maze[nr, nc] != self.WALL:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), path + [(nr, nc)]))
        
        return None
    
    def _augment(self, maze: np.ndarray, solution: np.ndarray, aug_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentation based on aug_id (0-7)."""
        # 8 augmentations: 4 rotations × 2 flips
        rotation = aug_id % 4
        flip = aug_id // 4
        
        m = maze.copy()
        s = solution.copy()
        
        # Rotate
        m = np.rot90(m, rotation)
        s = np.rot90(s, rotation)
        
        # Flip
        if flip:
            m = np.fliplr(m)
            s = np.fliplr(s)
        
        return m, s
    
    def __len__(self) -> int:
        return len(self.mazes) * self.n_augmentations
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        base_idx = idx // self.n_augmentations
        aug_id = idx % self.n_augmentations
        
        maze = self.mazes[base_idx]
        solution = self.solutions[base_idx]
        
        # Apply augmentation
        if aug_id > 0:
            maze, solution = self._augment(maze, solution, aug_id)
        
        # Flatten
        maze_flat = maze.flatten()
        solution_flat = solution.flatten()
        
        return (
            torch.tensor(maze_flat, dtype=torch.long),
            torch.tensor(solution_flat, dtype=torch.long)
        )
    
    @staticmethod
    def get_vocab_size() -> int:
        return MazeHardDataset.VOCAB_SIZE
    
    def get_context_length(self) -> int:
        return self.context_length
    
    def visualize(self, idx: int) -> str:
        """Visualize a maze and its solution."""
        base_idx = idx // self.n_augmentations
        maze = self.mazes[base_idx]
        solution = self.solutions[base_idx]
        
        # Character map
        chars = {
            self.EMPTY: '·',
            self.WALL: '█',
            self.START: 'S',
            self.END: 'E',
            self.PATH: '○'
        }
        
        def grid_to_str(grid):
            return '\n'.join(''.join(chars[c] for c in row) for row in grid)
        
        # Count path length
        path_length = np.sum(solution == self.PATH) + 2  # +2 for start and end
        
        return (
            f"Maze (path length: {path_length}):\n"
            f"{grid_to_str(maze)}\n\n"
            f"Solution:\n"
            f"{grid_to_str(solution)}"
        )


class MazeSimpleDataset(BaseDataset):
    """Simpler 10x10 mazes for faster experimentation."""
    
    EMPTY = 0
    WALL = 1
    START = 2
    END = 3
    PATH = 4
    
    VOCAB_SIZE = 5
    
    def __init__(
        self,
        n_samples: int = 500,
        grid_size: int = 10,
        wall_density: float = 0.25,
        min_path_length: int = 8,
        n_augmentations: int = 8,
        seed: Optional[int] = None
    ):
        # Reuse the hard dataset logic with simpler settings
        self._inner = MazeHardDataset(
            n_samples=n_samples,
            grid_size=grid_size,
            wall_density=wall_density,
            min_path_length=min_path_length,
            n_augmentations=n_augmentations,
            seed=seed
        )
        self.grid_size = grid_size
        self.context_length = grid_size * grid_size
        self.n_augmentations = n_augmentations
    
    def __len__(self):
        return len(self._inner)
    
    def __getitem__(self, idx):
        return self._inner[idx]
    
    @staticmethod
    def get_vocab_size():
        return MazeSimpleDataset.VOCAB_SIZE
    
    def get_context_length(self):
        return self.context_length
    
    def visualize(self, idx):
        return self._inner.visualize(idx)


if __name__ == "__main__":
    print("=" * 60)
    print("Maze-Hard Dataset Test")
    print("=" * 60)
    
    # Test small maze first
    print("\n--- Simple 10x10 Maze ---")
    simple = MazeSimpleDataset(n_samples=10, grid_size=10, seed=42)
    print(f"Dataset size: {len(simple)}")
    print(f"Vocab size: {simple.get_vocab_size()}")
    print(f"Context length: {simple.get_context_length()}")
    
    print("\nSample maze:")
    print(simple.visualize(0))
    
    x, y = simple[0]
    print(f"\nTensor shapes: input={x.shape}, target={y.shape}")
    
    # Test Maze-Hard (30x30)
    print("\n\n--- Maze-Hard 30x30 ---")
    hard = MazeHardDataset(n_samples=5, grid_size=30, min_path_length=20, seed=42)
    print(f"Dataset size: {len(hard)}")
    print(f"Context length: {hard.get_context_length()}")
    
    # Show first 10 rows of a sample
    print("\nSample 30x30 maze (first 15 rows):")
    viz = hard.visualize(0)
    lines = viz.split('\n')
    print('\n'.join(lines[:17]))  # Header + 15 rows
    print("...")
    
    print("\n✓ Maze datasets working!")
