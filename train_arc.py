#!/usr/bin/env python3
"""
Train TRM on ARC-AGI
====================

Train the Tiny Recursive Model on ARC-AGI tasks.
The paper achieved 44.6% on ARC-AGI-1 with this approach.

ARC tasks test abstract visual reasoning:
- Pattern transformations
- Object detection and manipulation
- Symmetry, rotation, scaling
- Color patterns and fills

Usage:
    python train_arc.py
    python train_arc.py --epochs 50 --batch-size 16
    
Note: ARC is MUCH harder than Sudoku. Even state-of-the-art
models struggle. The paper used extensive training (60K epochs!)
to achieve 44.6%. This script is for experimentation.
"""

import argparse
import torch
from torch.utils.data import DataLoader, random_split

import sys
sys.path.insert(0, '.')

from tiny_recursive_model.model import TinyRecursiveModel
from tiny_recursive_model.config import TRMConfig, TrainingConfig
from tiny_recursive_model.trainer import Trainer
from tiny_recursive_model.data.arc import ARCDataset, ARCFewShotDataset
from tiny_recursive_model.utils import set_seed, count_parameters, get_device


def main():
    parser = argparse.ArgumentParser(description="Train TRM on ARC-AGI")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--n-augmentations", type=int, default=8, help="Augmentations per sample")
    parser.add_argument("--max-grid-size", type=int, default=30, help="Max grid size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", type=str, default="standard", 
                        choices=["standard", "fewshot"],
                        help="Training mode: 'standard' treats each pair independently, "
                             "'fewshot' includes context examples")
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")
    
    # Create dataset based on mode
    print(f"\nLoading ARC-AGI dataset (mode: {args.mode})...")
    
    if args.mode == "standard":
        # Standard mode: each (input, output) pair is a sample
        full_dataset = ARCDataset(
            split='training',
            max_grid_size=args.max_grid_size,
            augment=True,
            n_augmentations=args.n_augmentations,
            include_examples=True,
            seed=args.seed
        )
        context_length = full_dataset.get_context_length()
        vocab_size = full_dataset.get_vocab_size()
        
    else:
        # Few-shot mode: include training examples as context
        full_dataset = ARCFewShotDataset(
            split='training',
            max_grid_size=15,  # Smaller for context length
            max_examples=2,
            seed=args.seed
        )
        context_length = full_dataset.get_context_length()
        vocab_size = full_dataset.get_vocab_size()
    
    # Split into train/test
    n_test = min(100, len(full_dataset) // 10)
    n_train = len(full_dataset) - n_test
    train_dataset, test_dataset = random_split(
        full_dataset, 
        [n_train, n_test],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Vocab size: {vocab_size}")
    print(f"Context length: {context_length}")
    
    # Show some examples
    if hasattr(full_dataset, 'visualize'):
        print("\n" + "=" * 50)
        print("Sample task:")
        print("=" * 50)
        print(full_dataset.visualize(0))
    
    # Create model
    # For ARC with 30x30 grids (900 tokens), use attention
    use_attention = context_length > 256
    
    model_config = TRMConfig(
        vocab_size=vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        num_heads=8 if use_attention else 4,
        context_length=context_length,
        mlp_ratio=4.0,
        n_recursions=6,
        T_steps=3,
        n_supervision=16,
        use_attention=use_attention
    )
    
    model = TinyRecursiveModel(model_config)
    print(f"\nModel parameters: {count_parameters(model):,}")
    print(f"Using attention: {use_attention}")
    print(f"Effective depth per step: {model_config.effective_depth}")
    print(f"Total effective depth: {model_config.total_depth}")
    
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
        weight_decay=0.1,
        warmup_steps=200,
        grad_clip=1.0,
        n_supervision=16,
        n_supervision_eval=16,
        use_ema=True,
        ema_decay=0.999,
        log_interval=20,
        eval_interval=100,
        save_interval=500,
        output_dir="outputs_arc",
        seed=args.seed
    )
    
    # Train
    print(f"\n" + "=" * 50)
    print(f"Training TRM on ARC-AGI")
    print(f"=" * 50)
    print(f"This is a HARD task - don't expect Sudoku-level accuracy!")
    print(f"The paper trained for 60K epochs to achieve 44.6%.")
    print(f"This run: {args.epochs} epochs for experimentation.\n")
    
    trainer = Trainer(model, train_config, device=device)
    results = trainer.train(train_loader, test_loader, model_config)
    
    # Analysis
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Final cell accuracy: {results['final_metrics']['cell_accuracy']:.2%}")
    print(f"Final puzzle accuracy: {results['final_metrics']['puzzle_accuracy']:.2%}")
    
    print("\nNote on ARC accuracy:")
    print("- Cell accuracy = % of individual pixels correct")
    print("- Puzzle accuracy = % of entire grids 100% correct")
    print("- ARC-AGI benchmark uses puzzle accuracy")
    print("- Even small puzzle accuracy is meaningful on ARC!")
    
    return results


if __name__ == "__main__":
    main()
