#!/usr/bin/env python3
"""
Train TRM on Maze-Hard
======================

30x30 maze pathfinding task from the TRM paper.
TRM achieves 85.3% accuracy on this benchmark.

Paper settings:
- 1000 mazes × 8 augmentations = 8000 samples
- 50,000 epochs (we use fewer for experimentation)
- Uses self-attention (not MLP-Mixer)
- H_cycles=3, L_cycles=4 (we call these T_steps and n_recursions)

Usage:
    # Quick test (~30 min)
    python train_maze.py --grid-size 10 --epochs 100 --n-train 200
    
    # Full Maze-Hard (~24 hours on L40S)
    python train_maze.py --grid-size 30 --epochs 5000 --n-train 1000
"""

import argparse
import torch
from torch.utils.data import DataLoader, random_split

import sys
sys.path.insert(0, '.')

from tiny_recursive_model.model import TinyRecursiveModel
from tiny_recursive_model.config import TRMConfig, TrainingConfig
from tiny_recursive_model.trainer import Trainer
from tiny_recursive_model.data.maze import MazeHardDataset, MazeSimpleDataset
from tiny_recursive_model.utils import set_seed, count_parameters, get_device


def main():
    parser = argparse.ArgumentParser(description="Train TRM on Maze")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs")
    parser.add_argument("--n-train", type=int, default=500, help="Number of base mazes")
    parser.add_argument("--n-augmentations", type=int, default=8, help="Augmentations per maze")
    parser.add_argument("--grid-size", type=int, default=15, help="Maze grid size (10=simple, 30=hard)")
    parser.add_argument("--min-path-length", type=int, default=10, help="Minimum path length")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")
    
    # Adjust difficulty based on grid size
    if args.grid_size <= 15:
        min_path = max(args.min_path_length, args.grid_size // 2)
        wall_density = 0.25
    else:
        min_path = max(args.min_path_length, args.grid_size * 2 // 3)
        wall_density = 0.3
    
    # Create dataset
    print(f"\nGenerating {args.n_train} mazes ({args.grid_size}x{args.grid_size})...")
    
    full_dataset = MazeHardDataset(
        n_samples=args.n_train,
        grid_size=args.grid_size,
        wall_density=wall_density,
        min_path_length=min_path,
        n_augmentations=args.n_augmentations,
        seed=args.seed
    )
    
    context_length = full_dataset.get_context_length()
    vocab_size = full_dataset.get_vocab_size()
    
    # Split into train/test
    n_test = min(100 * args.n_augmentations, len(full_dataset) // 10)
    n_train = len(full_dataset) - n_test
    
    train_dataset, test_dataset = random_split(
        full_dataset,
        [n_train, n_test],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"\nDataset:")
    print(f"  Base mazes: {len(full_dataset.mazes)}")
    print(f"  Total samples: {len(full_dataset)} ({args.n_augmentations}× augmentation)")
    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    print(f"  Grid size: {args.grid_size}×{args.grid_size} = {context_length} tokens")
    print(f"  Vocab size: {vocab_size}")
    
    # Show sample
    print("\n" + "=" * 50)
    print("Sample maze:")
    print("=" * 50)
    viz = full_dataset.visualize(0)
    lines = viz.split('\n')
    # Show first 12 lines (or all if small)
    for line in lines[:min(14, len(lines))]:
        print(line)
    if len(lines) > 14:
        print("...")
    
    # Create model - use attention for variable/large grids
    use_attention = args.grid_size > 15
    
    model_config = TRMConfig(
        vocab_size=vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        num_heads=8 if use_attention else 4,
        context_length=context_length,
        mlp_ratio=4.0,
        n_recursions=4,  # L_cycles in paper
        T_steps=3,       # H_cycles in paper
        n_supervision=16,
        use_attention=use_attention
    )
    
    model = TinyRecursiveModel(model_config)
    print(f"\nModel:")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Using attention: {use_attention}")
    print(f"  Effective depth: {model_config.total_depth}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Training config
    train_config = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=1.0,  # Paper uses 1.0 for maze
        warmup_steps=100,
        grad_clip=1.0,
        n_supervision=16,
        n_supervision_eval=16,
        use_ema=True,
        ema_decay=0.999,
        log_interval=50,
        eval_interval=200,
        save_interval=1000,
        output_dir=f"outputs_maze_{args.grid_size}x{args.grid_size}",
        seed=args.seed
    )
    
    # Train
    print(f"\n" + "=" * 50)
    print(f"Training TRM on Maze-{args.grid_size}x{args.grid_size}")
    print(f"=" * 50)
    
    trainer = Trainer(model, train_config, device=device)
    results = trainer.train(train_loader, test_loader, model_config)
    
    # Final visualization
    print("\n" + "=" * 50)
    print("Testing on sample mazes:")
    print("=" * 50)
    
    model.eval()
    model = model.to(device)
    
    correct = 0
    total = 5
    
    for i in range(total):
        x, y = test_dataset[i]
        x_batch = x.unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model.predict(x_batch, n_supervision=16)
        
        pred = pred[0].cpu()
        
        # Calculate accuracy
        # Only count path cells (ignore walls which are given)
        path_mask = (y == 4)  # PATH token
        if path_mask.sum() > 0:
            path_correct = (pred[path_mask] == y[path_mask]).float().mean().item()
        else:
            path_correct = 1.0
        
        full_correct = (pred == y).all().item()
        correct += int(full_correct)
        
        status = "✓" if full_correct else f"~{path_correct:.0%}"
        print(f"  Maze {i+1}: {status}")
    
    print(f"\nPerfect mazes: {correct}/{total}")
    print(f"Final cell accuracy: {results['final_metrics']['cell_accuracy']:.2%}")
    print(f"Final puzzle accuracy: {results['final_metrics']['puzzle_accuracy']:.2%}")
    
    print("\nNote: Paper reports 85.3% on Maze-Hard (30×30)")
    print("This requires ~50,000 epochs and more training time.")
    
    return results


if __name__ == "__main__":
    main()
