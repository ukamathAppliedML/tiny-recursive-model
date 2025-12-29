#!/usr/bin/env python3
"""
Train TRM on Arithmetic Task
============================

A simpler task than Sudoku - learns 3-digit addition.
Trains much faster and demonstrates recursive reasoning on carry propagation.

Usage:
    python train_arithmetic.py
    python train_arithmetic.py --epochs 20 --n-train 2000
"""

import argparse
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, '.')

from tiny_recursive_model.model import TinyRecursiveModel
from tiny_recursive_model.config import TRMConfig, TrainingConfig
from tiny_recursive_model.trainer import Trainer
from tiny_recursive_model.data.arithmetic import ArithmeticDataset
from tiny_recursive_model.utils import set_seed, count_parameters, get_device


def main():
    parser = argparse.ArgumentParser(description="Train TRM on arithmetic")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--n-train", type=int, default=5000, help="Training samples")
    parser.add_argument("--n-test", type=int, default=500, help="Test samples")
    parser.add_argument("--n-digits", type=int, default=3, help="Digits per number")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = ArithmeticDataset(
        n_samples=args.n_train,
        n_digits=args.n_digits,
        augment=True,
        seed=args.seed
    )
    test_dataset = ArithmeticDataset(
        n_samples=args.n_test,
        n_digits=args.n_digits,
        augment=False,
        seed=args.seed + 1000
    )
    
    context_length = train_dataset.get_context_length()
    vocab_size = train_dataset.get_vocab_size()
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Vocab size: {vocab_size}")
    print(f"Context length: {context_length}")
    
    # Show some examples
    print("\nSample problems:")
    for i in range(3):
        print(f"  {train_dataset.visualize(i)}")
    
    # Create model - smaller than Sudoku since task is simpler
    model_config = TRMConfig(
        vocab_size=vocab_size,
        hidden_dim=128,  # Smaller for this simpler task
        num_layers=2,
        num_heads=4,
        context_length=context_length,
        mlp_ratio=4.0,
        n_recursions=6,
        T_steps=3,
        n_supervision=16,
        use_attention=False  # MLP-Mixer
    )
    
    model = TinyRecursiveModel(model_config)
    print(f"\nModel parameters: {count_parameters(model):,}")
    
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
        warmup_steps=100,
        grad_clip=1.0,
        n_supervision=16,
        n_supervision_eval=16,
        use_ema=True,
        ema_decay=0.999,
        log_interval=50,
        eval_interval=200,
        save_interval=500,
        output_dir="outputs_arithmetic",
        seed=args.seed
    )
    
    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    trainer = Trainer(model, train_config, device=device)
    results = trainer.train(train_loader, test_loader, model_config)
    
    # Final test with detailed output
    print("\n" + "=" * 50)
    print("Testing on specific examples:")
    print("=" * 50)
    
    model.eval()
    model = model.to(device)
    
    # Test on a few specific additions
    test_problems = [
        (123, 456),
        (999, 1),
        (250, 250),
        (111, 222),
        (0, 500),
    ]
    
    n = args.n_digits
    
    for num1, num2 in test_problems:
        if num1 >= 10**n or num2 >= 10**n:
            continue
            
        expected = num1 + num2
        if expected >= 10**n:
            continue
        
        # Create input
        def to_digits(x):
            return [int(d) for d in str(x).zfill(n)]
        
        d1 = to_digits(num1)
        d2 = to_digits(num2)
        
        input_seq = d1 + [10] + d2 + [11] + [12] * n
        input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)
        
        # Predict
        with torch.no_grad():
            pred = model.predict(input_tensor, n_supervision=16)
        
        pred_result_digits = pred[0, -n:].cpu().tolist()
        pred_result = int(''.join(str(d) for d in pred_result_digits))
        
        status = "✓" if pred_result == expected else "✗"
        print(f"  {num1} + {num2} = {pred_result} (expected {expected}) {status}")
    
    print("\nDone!")
    return results


if __name__ == "__main__":
    main()
