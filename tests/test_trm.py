"""
Tests for Tiny Recursive Model
==============================

Run with: pytest tests/
"""

import pytest
import torch
import numpy as np

from tiny_recursive_model import TinyRecursiveModel, TRMConfig
from tiny_recursive_model.data.sudoku import (
    generate_complete_sudoku,
    generate_puzzle,
    solve_sudoku,
    is_valid_sudoku,
    is_complete_sudoku,
    augment_sudoku,
    SudokuDataset,
    SimpleSudokuDataset
)
from tiny_recursive_model.utils import count_parameters


class TestTRMConfig:
    """Tests for TRMConfig."""
    
    def test_default_config(self):
        config = TRMConfig()
        assert config.vocab_size == 10
        assert config.hidden_dim == 256
        assert config.num_layers == 2
        assert config.context_length == 81
    
    def test_effective_depth(self):
        config = TRMConfig(
            num_layers=2,
            n_recursions=6,
            T_steps=3,
            n_supervision=16
        )
        # 3 * (6+1) * 2 = 42
        assert config.effective_depth == 42
        # 16 * 42 = 672
        assert config.total_depth == 672
    
    def test_to_dict(self):
        config = TRMConfig(vocab_size=20)
        d = config.to_dict()
        assert d["vocab_size"] == 20
        assert isinstance(d, dict)
    
    def test_from_dict(self):
        d = {"vocab_size": 20, "hidden_dim": 512}
        config = TRMConfig.from_dict(d)
        assert config.vocab_size == 20
        assert config.hidden_dim == 512


class TestTinyRecursiveModel:
    """Tests for TinyRecursiveModel."""
    
    @pytest.fixture
    def model(self):
        config = TRMConfig(
            vocab_size=10,
            hidden_dim=64,  # Small for testing
            num_layers=2,
            context_length=81,
            n_recursions=2,
            T_steps=2,
            n_supervision=4
        )
        return TinyRecursiveModel(config)
    
    def test_forward_shape(self, model):
        batch_size = 4
        x = torch.randint(0, 10, (batch_size, 81))
        
        (y, z), logits, q_hat = model(x)
        
        assert y.shape == (batch_size, 81, model.config.hidden_dim)
        assert z.shape == (batch_size, 81, model.config.hidden_dim)
        assert logits.shape == (batch_size, 81, 10)
        assert q_hat.shape == (batch_size,)
    
    def test_forward_with_state(self, model):
        batch_size = 4
        x = torch.randint(0, 10, (batch_size, 81))
        
        # First pass
        (y1, z1), _, _ = model(x)
        
        # Second pass with state
        (y2, z2), logits, _ = model(x, y1, z1)
        
        assert logits.shape == (batch_size, 81, 10)
    
    def test_predict(self, model):
        batch_size = 4
        x = torch.randint(0, 10, (batch_size, 81))
        
        pred = model.predict(x, n_supervision=2)
        
        assert pred.shape == (batch_size, 81)
        assert pred.dtype == torch.long
    
    def test_parameter_count(self, model):
        n_params = count_parameters(model)
        assert n_params > 0
        assert n_params == model.get_num_params()


class TestSudokuGeneration:
    """Tests for Sudoku generation utilities."""
    
    def test_generate_complete_sudoku(self):
        grid = generate_complete_sudoku()
        assert grid.shape == (9, 9)
        assert is_complete_sudoku(grid)
    
    def test_generate_puzzle(self):
        solution = generate_complete_sudoku()
        puzzle = generate_puzzle(solution, n_hints=25)
        
        assert puzzle.shape == (9, 9)
        assert (puzzle == 0).sum() == 81 - 25
        assert is_valid_sudoku(puzzle)
    
    def test_solve_sudoku(self):
        solution = generate_complete_sudoku()
        puzzle = generate_puzzle(solution, n_hints=30)
        
        solved = solve_sudoku(puzzle)
        
        assert solved is not None
        assert is_complete_sudoku(solved)
        assert np.array_equal(solved, solution)
    
    def test_augment_preserves_validity(self):
        solution = generate_complete_sudoku()
        puzzle = generate_puzzle(solution, n_hints=25)
        
        aug_puzzle, aug_solution = augment_sudoku(puzzle, solution)
        
        assert is_valid_sudoku(aug_puzzle)
        assert is_complete_sudoku(aug_solution)


class TestSudokuDataset:
    """Tests for SudokuDataset."""
    
    def test_simple_dataset(self):
        dataset = SimpleSudokuDataset(n_samples=10)
        
        assert len(dataset) == 10
        
        puzzle, solution = dataset[0]
        assert puzzle.shape == (81,)
        assert solution.shape == (81,)
        assert puzzle.dtype == torch.long
    
    def test_augmented_dataset(self):
        dataset = SudokuDataset(
            n_puzzles=5,
            n_augmentations=10
        )
        
        assert len(dataset) == 50  # 5 * 10
        
        puzzle, solution = dataset[0]
        assert puzzle.shape == (81,)
    
    def test_vocab_and_context(self):
        assert SudokuDataset.get_vocab_size() == 10
        assert SudokuDataset.get_context_length() == 81


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
