"""
Demo Module
===========

Interactive demonstration of TRM architecture and capabilities.
"""

import torch
import numpy as np
import time
from typing import Optional
from pathlib import Path

from .model import TinyRecursiveModel
from .config import TRMConfig
from .data.sudoku import (
    generate_complete_sudoku, 
    generate_puzzle, 
    solve_sudoku,
    visualize_sudoku
)
from .utils import count_parameters, get_device, load_checkpoint


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def demo_architecture():
    """Demonstrate TRM architecture."""
    print_header("1. TRM Architecture Overview")
    
    config = TRMConfig(
        vocab_size=10,
        hidden_dim=256,
        num_layers=2,
        num_heads=4,
        context_length=81,
        n_recursions=6,
        T_steps=3,
        n_supervision=16,
        use_attention=False
    )
    
    model = TinyRecursiveModel(config)
    n_params = count_parameters(model)
    
    print(f"\nModel Configuration:")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Hidden dimension: {config.hidden_dim}")
    print(f"  Number of layers: {config.num_layers}")
    print(f"  Context length: {config.context_length}")
    print(f"  Token mixing: {'Attention' if config.use_attention else 'MLP-Mixer'}")
    
    print(f"\nRecursion Parameters:")
    print(f"  n (recursions per step): {config.n_recursions}")
    print(f"  T (gradient-free steps): {config.T_steps}")
    print(f"  N_sup (max supervision steps): {config.n_supervision}")
    
    print(f"\nEffective Depth Analysis:")
    print(f"  Per supervision step: {config.effective_depth} layers")
    print(f"  Total (all steps): {config.total_depth} layers")
    print(f"  → A 2-layer network acts like a {config.total_depth}-layer network!")
    
    print(f"\nParameter Count: {n_params:,} ({n_params/1e6:.2f}M)")
    
    return model, config


def demo_forward_pass(model: TinyRecursiveModel):
    """Demonstrate forward pass."""
    print_header("2. Forward Pass Demonstration")
    
    device = get_device()
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Create puzzle
    solution = generate_complete_sudoku()
    puzzle = generate_puzzle(solution, n_hints=30)
    
    print(visualize_sudoku(puzzle, "\nSample Puzzle (30 hints):"))
    
    # Convert to tensor
    puzzle_tensor = torch.tensor(
        puzzle.flatten(), dtype=torch.long
    ).unsqueeze(0).to(device)
    
    print(f"\nInput tensor shape: {puzzle_tensor.shape}")
    
    # Forward pass
    print("\n--- First Supervision Step ---")
    (y, z), logits, q_hat = model(puzzle_tensor)
    
    print(f"  y (answer embedding) shape: {y.shape}")
    print(f"  z (latent state) shape: {z.shape}")
    print(f"  logits shape: {logits.shape}")
    print(f"  q_hat (halt probability logit): {q_hat.item():.4f}")
    
    # Accuracy
    pred = logits.argmax(dim=-1).squeeze(0)
    solution_flat = torch.tensor(solution.flatten()).to(device)
    accuracy = (pred == solution_flat).float().mean().item()
    print(f"\n  Cell accuracy after 1 step: {accuracy:.1%}")
    
    # Second step
    print("\n--- Second Supervision Step ---")
    (y, z), logits, q_hat = model(puzzle_tensor, y, z)
    
    pred = logits.argmax(dim=-1).squeeze(0)
    accuracy = (pred == solution_flat).float().mean().item()
    print(f"  Cell accuracy after 2 steps: {accuracy:.1%}")
    
    return device


def demo_deep_supervision(model: TinyRecursiveModel, device: torch.device):
    """Demonstrate deep supervision."""
    print_header("3. Deep Supervision: Iterative Improvement")
    
    solution = generate_complete_sudoku()
    puzzle = generate_puzzle(solution, n_hints=25)
    
    puzzle_tensor = torch.tensor(
        puzzle.flatten(), dtype=torch.long
    ).unsqueeze(0).to(device)
    solution_tensor = torch.tensor(solution.flatten(), dtype=torch.long).to(device)
    
    print("\nRunning 16 supervision steps...")
    print("(Note: With untrained model, improvement is limited)\n")
    
    y, z = None, None
    accuracies = []
    
    with torch.no_grad():
        for step in range(16):
            (y, z), logits, q_hat = model(puzzle_tensor, y, z)
            
            pred = logits.argmax(dim=-1).squeeze(0)
            acc = (pred == solution_tensor).float().mean().item()
            accuracies.append(acc)
            
            halt_prob = torch.sigmoid(q_hat).item()
            
            if step < 4 or step >= 12:
                print(f"  Step {step+1:2d}: Accuracy = {acc:.1%}, Halt prob = {halt_prob:.2%}")
            elif step == 4:
                print("  ...")
    
    print(f"\nAccuracy progression:")
    print(f"  Step 1:  {accuracies[0]:.1%}")
    print(f"  Step 8:  {accuracies[7]:.1%}")
    print(f"  Step 16: {accuracies[15]:.1%}")
    
    # Visualization
    print("\nAccuracy over steps:")
    max_bar = 40
    for i, acc in enumerate(accuracies):
        bar_len = int(acc * max_bar)
        bar = "█" * bar_len + "░" * (max_bar - bar_len)
        print(f"  {i+1:2d} |{bar}| {acc:.0%}")


def demo_inference_speed(model: TinyRecursiveModel, device: torch.device):
    """Benchmark inference speed."""
    print_header("4. Inference Speed Benchmark")
    
    model.eval()
    
    batch_sizes = [1, 4, 8, 16]
    n_iterations = 20
    
    print(f"Device: {device}")
    print(f"Iterations per measurement: {n_iterations}")
    print(f"Supervision steps per inference: 16\n")
    
    for batch_size in batch_sizes:
        x = torch.randint(0, 10, (batch_size, 81)).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model.predict(x, n_supervision=16)
        
        # Synchronize
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        with torch.no_grad():
            for _ in range(n_iterations):
                _ = model.predict(x, n_supervision=16)
        
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        
        samples_per_sec = (batch_size * n_iterations) / elapsed
        ms_per_sample = (elapsed / (batch_size * n_iterations)) * 1000
        
        print(f"  Batch {batch_size:2d}: {samples_per_sec:6.1f} samples/sec | {ms_per_sample:6.2f} ms/sample")


def demo_comparison():
    """Compare TRM to traditional approaches."""
    print_header("5. Why TRM Works")
    
    print("""
    Traditional Transformer:
    ────────────────────────
    • Stack many layers (12-96 layers)
    • Each layer used once
    • Memory scales with depth
    • Parameters: 100M - 100B+
    
    Tiny Recursive Model:
    ─────────────────────
    • Just 2 layers!
    • Each layer reused 336 times
    • Memory efficient (only 2 layers)
    • Parameters: ~5-7M
    
    The Trade-off:
    ─────────────
    • TRM is slower (sequential)
    • But uses far less memory
    • And generalizes better on small data!
    
    Why 2 Layers is Optimal:
    ────────────────────────
    • More layers → overfitting on small data
    • 2 layers with recursion = depth without overfitting
    • Recursion provides depth; network provides flexibility
    """)


def run_demo(checkpoint_path: Optional[str] = None):
    """Run the complete demo.
    
    Parameters
    ----------
    checkpoint_path : str, optional
        Path to trained model checkpoint
    """
    print("\n" + "=" * 60)
    print("   TINY RECURSIVE MODEL - DEMO")
    print("   Based on 'Less is More' (arXiv:2510.04871)")
    print("=" * 60)
    
    # Load or create model
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"\nLoading model from {checkpoint_path}...")
        device = get_device()
        checkpoint = load_checkpoint(checkpoint_path)
        config_dict = checkpoint.get("config", {})
        config = TRMConfig(**config_dict) if config_dict else TRMConfig()
        model = TinyRecursiveModel(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model with {count_parameters(model):,} parameters")
    else:
        model, config = demo_architecture()
        device = demo_forward_pass(model)
    
    demo_deep_supervision(model, device)
    demo_inference_speed(model, device)
    demo_comparison()
    
    print_header("Demo Complete!")
    print("""
    Next Steps:
    ───────────
    1. Train: trm train --task sudoku --epochs 50
    2. Or:    python -m tiny_recursive_model.train
    3. Read:  tiny_recursive_model/model.py
    """)


if __name__ == "__main__":
    run_demo()
