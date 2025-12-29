"""
Command Line Interface
======================

CLI for training and running TRM models.

Usage:
    trm train --task sudoku --epochs 50
    trm demo
    trm predict --checkpoint model.pt --input puzzle.txt
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="trm",
        description="Tiny Recursive Model - A minimal recursive reasoning network"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a TRM model")
    train_parser.add_argument(
        "--task", type=str, default="sudoku",
        choices=["sudoku", "maze", "arc"],
        help="Task to train on"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size"
    )
    train_parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate"
    )
    train_parser.add_argument(
        "--n-train", type=int, default=500,
        help="Number of training samples"
    )
    train_parser.add_argument(
        "--n-test", type=int, default=100,
        help="Number of test samples"
    )
    train_parser.add_argument(
        "--difficulty", type=str, default="medium",
        choices=["easy", "medium", "hard"],
        help="Puzzle difficulty"
    )
    train_parser.add_argument(
        "--output", type=str, default="outputs",
        help="Output directory"
    )
    train_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run architecture demo")
    demo_parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint"
    )
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    predict_parser.add_argument(
        "--input", type=str, required=True,
        help="Input file or puzzle string"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    if args.command == "train":
        run_train(args)
    elif args.command == "demo":
        run_demo(args)
    elif args.command == "predict":
        run_predict(args)


def run_train(args):
    """Run training."""
    from .trainer import train
    from .config import TrainingConfig
    
    config = TrainingConfig(
        n_train_samples=args.n_train,
        n_test_samples=args.n_test,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output,
        seed=args.seed
    )
    
    train(
        task=args.task,
        train_config=config,
        difficulty=args.difficulty
    )


def run_demo(args):
    """Run demo."""
    from .demo import run_demo as demo_main
    demo_main(checkpoint_path=args.checkpoint)


def run_predict(args):
    """Run prediction."""
    import torch
    from .model import TinyRecursiveModel
    from .config import TRMConfig
    from .utils import load_checkpoint, get_device
    
    device = get_device()
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint)
    config_dict = checkpoint.get("config", {})
    config = TRMConfig(**config_dict) if config_dict else TRMConfig()
    
    model = TinyRecursiveModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Parse input
    input_path = Path(args.input)
    if input_path.exists():
        with open(input_path) as f:
            input_str = f.read().strip()
    else:
        input_str = args.input
    
    # Convert to tensor
    tokens = [int(c) if c.isdigit() else 0 for c in input_str.replace(".", "0")]
    tokens = tokens[:config.context_length]
    tokens += [0] * (config.context_length - len(tokens))
    
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        pred = model.predict(input_tensor)
    
    # Print result
    pred_np = pred.squeeze().cpu().numpy()
    print("Prediction:")
    print(pred_np.reshape(9, 9) if len(pred_np) == 81 else pred_np)


if __name__ == "__main__":
    main()
