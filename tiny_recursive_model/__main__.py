"""
Main entry point for running TRM as a module.

Usage:
    python -m tiny_recursive_model demo      # Run demo
    python -m tiny_recursive_model train     # Train model
    python -m tiny_recursive_model --help    # Show help
"""

import sys

def main():
    if len(sys.argv) < 2:
        print("Tiny Recursive Model")
        print("=" * 40)
        print("\nUsage:")
        print("  python -m tiny_recursive_model demo     Run interactive demo")
        print("  python -m tiny_recursive_model train    Train a model")
        print("\nFor training options:")
        print("  python -m tiny_recursive_model train --help")
        return
    
    command = sys.argv[1]
    
    if command == "demo":
        from .demo import run_demo
        run_demo()
    
    elif command == "train":
        # Remove 'train' from argv so argparse works correctly
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        
        import argparse
        parser = argparse.ArgumentParser(
            prog="python -m tiny_recursive_model train",
            description="Train a Tiny Recursive Model"
        )
        parser.add_argument(
            "--task", type=str, default="sudoku",
            choices=["sudoku"],
            help="Task to train on (default: sudoku)"
        )
        parser.add_argument(
            "--epochs", type=int, default=50,
            help="Number of training epochs (default: 50)"
        )
        parser.add_argument(
            "--batch-size", type=int, default=32,
            help="Batch size (default: 32)"
        )
        parser.add_argument(
            "--lr", type=float, default=1e-4,
            help="Learning rate (default: 1e-4)"
        )
        parser.add_argument(
            "--n-train", type=int, default=500,
            help="Number of training puzzles (default: 500)"
        )
        parser.add_argument(
            "--n-test", type=int, default=100,
            help="Number of test puzzles (default: 100)"
        )
        parser.add_argument(
            "--n-augmentations", type=int, default=100,
            help="Augmentations per puzzle (default: 100)"
        )
        parser.add_argument(
            "--n-supervision", type=int, default=16,
            help="Deep supervision steps during training (default: 16)"
        )
        parser.add_argument(
            "--difficulty", type=str, default="medium",
            choices=["easy", "medium", "hard"],
            help="Puzzle difficulty (default: medium)"
        )
        parser.add_argument(
            "--output", type=str, default="outputs",
            help="Output directory (default: outputs)"
        )
        parser.add_argument(
            "--seed", type=int, default=42,
            help="Random seed (default: 42)"
        )
        
        args = parser.parse_args()
        
        from .trainer import train
        from .config import TrainingConfig
        
        config = TrainingConfig(
            n_train_samples=args.n_train,
            n_test_samples=args.n_test,
            n_augmentations=args.n_augmentations,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            n_supervision=args.n_supervision,
            n_supervision_eval=args.n_supervision,
            output_dir=args.output,
            seed=args.seed
        )
        
        train(
            task=args.task,
            train_config=config,
            difficulty=args.difficulty
        )
    
    elif command in ["--help", "-h", "help"]:
        print("Tiny Recursive Model")
        print("=" * 40)
        print("\nUsage:")
        print("  python -m tiny_recursive_model demo     Run interactive demo")
        print("  python -m tiny_recursive_model train    Train a model")
        print("\nFor training options:")
        print("  python -m tiny_recursive_model train --help")
    
    else:
        print(f"Unknown command: {command}")
        print("Use 'demo' or 'train'")
        sys.exit(1)


if __name__ == "__main__":
    main()
