# Tiny Recursive Model (TRM)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A clean, educational implementation of the **Tiny Recursive Model** from Samsung SAIL Montreal's paper ["Less is More: Recursive Reasoning with Tiny Networks"](https://arxiv.org/abs/2510.04871).

TRM achieves remarkable results on complex reasoning tasks with only **7M parameters**—outperforming billion-parameter LLMs on benchmarks like ARC-AGI, Sudoku, and Maze pathfinding.

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Benchmarks](#benchmarks)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Training Your Own Models](#training-your-own-models)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## Overview

### What is TRM?

Tiny Recursive Model (TRM) is a **recursive reasoning architecture** that solves complex puzzles by iteratively refining its answers. Instead of scaling up model size, TRM achieves depth through recursion—running the same small network multiple times to progressively improve solutions.

### Why TRM Matters

| Approach | Parameters | ARC-AGI-1 | Cost |
|----------|------------|-----------|------|
| **TRM** | **7M** | **44.6%** | **~$500** |
| Gemini 2.5 Pro | ~1T+ | 37.0% | $$$$ |
| o3-mini-high | ~100B+ | 34.5% | $$$$ |
| DeepSeek-R1 | 671B | 15.8% | $$$$ |

TRM demonstrates that **architectural innovation can outperform brute-force scaling** on structured reasoning tasks.

### Core Insight

> "The idea that one must rely on massive foundational models trained for millions of dollars by some big corporation in order to achieve success on hard tasks is a trap."
> — Alexia Jolicoeur-Martineau, TRM Paper

---

## Key Results

Results from the original paper (7M parameter model):

| Benchmark | TRM Score | Previous SOTA | Task Type |
|-----------|-----------|---------------|-----------|
| **Sudoku-Extreme** | 87.4% | 55.0% (HRM) | Constraint satisfaction |
| **Maze-Hard** | 85.3% | 74.5% (HRM) | Pathfinding (30×30) |
| **ARC-AGI-1** | 44.6% | 40.3% (HRM) | Abstract reasoning |
| **ARC-AGI-2** | 7.8% | 5.0% (HRM) | Abstract reasoning (hard) |

---

## Architecture

### How TRM Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRM Recursive Loop                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input: x (question), y₀ (initial guess), z₀ (latent state)   │
│                                                                 │
│   For k = 1 to K supervision steps:                            │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  "Think" Phase: Update latent n times                   │   │
│   │  ┌─────────────────────────────────────────────────┐    │   │
│   │  │  for i = 1 to n:                                │    │   │
│   │  │      z ← f(x, y, z)   # Recursive reasoning     │    │   │
│   │  └─────────────────────────────────────────────────┘    │   │
│   │                                                         │   │
│   │  "Act" Phase: Update answer                             │   │
│   │  ┌─────────────────────────────────────────────────┐    │   │
│   │  │  y ← g(y, z)          # Refine prediction       │    │   │
│   │  └─────────────────────────────────────────────────┘    │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   Output: yₖ (refined answer)                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Single Tiny Network**: 2-layer transformer (or MLP-Mixer for fixed grids)
2. **Dual State**: Maintains both answer `y` and latent reasoning `z`
3. **Deep Supervision**: Loss computed at every refinement step
4. **Full Backpropagation**: Gradients flow through entire recursion (no fixed-point approximation)

### Mathematical Formulation

The TRM forward pass can be written as:

```
Initialize: y₀ = embed(input), z₀ = zeros

For each supervision step k = 1 to K:
    # Think phase: recursive latent updates
    For i = 1 to n:
        z ← LayerNorm(z + f([x; y; z]))
    
    # Act phase: update answer
    y ← LayerNorm(y + g([y; z]))
    
    # Compute loss at this step
    logits_k = output_head(y)
    loss_k = CrossEntropy(logits_k, target)

Total loss = mean(loss_1, loss_2, ..., loss_K)
```

### Effective Depth

TRM achieves depth through recursion rather than stacking layers:

```
Effective Depth = T_steps × (n_recursions + 1) × num_layers
                = 3 × (6 + 1) × 2
                = 42 effective layers (with only 2 actual layers!)
```

### Architecture Variants

| Variant | Token Mixer | Best For | Parameters |
|---------|-------------|----------|------------|
| **TRM-MLP** | MLP-Mixer | Fixed grids (Sudoku 9×9) | ~5M |
| **TRM-Attention** | Self-Attention | Variable grids (ARC, Maze) | ~7M |

### Why It Works

1. **Iterative Refinement**: Like a human solving Sudoku—make an initial guess, check for errors, refine
2. **Deep Supervision**: Every step contributes to learning, not just the final output
3. **Parameter Efficiency**: Shared weights across recursions prevent overfitting
4. **Implicit Error Correction**: The model learns to detect and fix its own mistakes

---

## Installation

### Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.9 | 3.10+ |
| RAM | 8GB | 16GB+ |
| Storage | 500MB | 2GB |
| GPU | Optional | CUDA/MPS/ROCm |

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/tiny-recursive-model.git
cd tiny-recursive-model

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows CMD:
venv\Scripts\activate.bat
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# 4. Install PyTorch (choose your platform)
```

#### PyTorch Installation by Platform

**macOS (Apple Silicon M1/M2/M3/M4):**
```bash
pip install torch torchvision torchaudio
```

**macOS (Intel) / Linux (CPU only):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Linux/Windows (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Linux/Windows (CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Complete Installation

```bash
# 5. Install TRM package
pip install -e .

# 6. Verify installation
python -c "from tiny_recursive_model import TinyRecursiveModel; print('✓ TRM installed!')"

# 7. Run quick demo
python -m tiny_recursive_model demo
```

---

## Quick Start

### Option 1: Interactive Demo (30 seconds)

```bash
python -m tiny_recursive_model demo
```

Shows TRM solving Sudoku puzzles with visualization of the recursive refinement process.

### Option 2: Train on Sudoku (2-5 hours)

```bash
# Quick training (~2 hours on M4 Mac)
python -m tiny_recursive_model train \
    --epochs 20 \
    --n-train 400 \
    --n-augmentations 100

# Full training (matches paper, ~5 hours)
python -m tiny_recursive_model train \
    --epochs 30 \
    --n-train 500 \
    --n-augmentations 150 \
    --batch-size 32
```

### Option 3: Train on Maze (30 min - 24 hours)

```bash
# Quick test (10×10 mazes, ~30 min)
python train_maze.py --grid-size 10 --epochs 200 --n-train 200

# Full Maze-Hard (30×30, ~24 hours)
python train_maze.py --grid-size 30 --epochs 5000 --n-train 1000
```

### Option 4: Train on Arithmetic (20 min)

```bash
# Simplest task - good for testing the setup
python train_arithmetic.py --epochs 30 --n-train 5000
```

### Option 5: Python API

```python
import torch
from tiny_recursive_model import TinyRecursiveModel, TRMConfig

# Create model
config = TRMConfig(
    vocab_size=10,
    hidden_dim=256,
    num_layers=2,
    num_heads=8,
    context_length=81,  # 9×9 Sudoku
    n_recursions=6,
    T_steps=3,
    n_supervision=16,
    use_attention=False  # MLP-Mixer for fixed grids
)

model = TinyRecursiveModel(config)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Forward pass with deep supervision
x = torch.randint(0, 10, (1, 81))  # Input puzzle
y = torch.randint(0, 10, (1, 81))  # Target solution

outputs = model(x, y, n_supervision=16)
print(f"Loss: {outputs['loss'].item():.4f}")
print(f"Supervision steps: {len(outputs['logits_list'])}")

# Inference (no target needed)
with torch.no_grad():
    prediction = model.predict(x, n_supervision=16)
print(f"Prediction shape: {prediction.shape}")
```

---

## Benchmarks

### Benchmark Comparison

| Task | Complexity | Grid Size | Training Time* | Script |
|------|------------|-----------|----------------|--------|
| **Arithmetic** | Easy | Variable | ~20 min | `train_arithmetic.py` |
| **Maze-Simple** | Easy | 10×10 | ~30 min | `train_maze.py --grid-size 10` |
| **Sudoku** | Medium | 9×9 | ~5 hours | `python -m tiny_recursive_model train` |
| **Maze-Hard** | Hard | 30×30 | ~24 hours | `train_maze.py --grid-size 30` |
| **ARC-AGI** | Very Hard | ≤30×30 | ~3 days | `train_arc.py` |

*Times are approximate for M4 Mac or equivalent GPU

### Sudoku-Extreme

9×9 Sudoku puzzles with minimal given cells (17-25 clues).

```bash
python -m tiny_recursive_model train \
    --epochs 30 \
    --n-train 500 \
    --n-augmentations 150 \
    --batch-size 32
```

**Expected Results:**
- Cell accuracy: 85-90%
- Puzzle accuracy: 15-25%
- Paper reports: 87.4% puzzle accuracy (with 50K epochs)

### Maze-Hard

30×30 grid maze with pathfinding. Model must find the shortest path from start to end.

```bash
# Quick test
python train_maze.py --grid-size 10 --epochs 200 --n-train 200

# Full benchmark
python train_maze.py --grid-size 30 --epochs 5000 --n-train 1000
```

**Expected Results:**
- Paper reports: 85.3% on 30×30 mazes
- Requires self-attention (not MLP-Mixer)

### ARC-AGI

Abstract Reasoning Corpus - visual pattern recognition and transformation.

```bash
python train_arc.py --epochs 50 --batch-size 16
```

**Notes:**
- Automatically downloads ARC dataset from GitHub
- Paper trained for 100K epochs on 4× H100 GPUs (~3 days)
- This is the hardest benchmark - even 5-10% is meaningful

### Arithmetic

3-digit addition - a simple task for validating the training pipeline.

```bash
python train_arithmetic.py --epochs 30 --n-train 5000
```

**Expected Results:**
- Should achieve 90%+ accuracy quickly
- Good for testing your setup before longer training runs

---

## Project Structure

```
tiny-recursive-model/
├── tiny_recursive_model/          # Main package
│   ├── __init__.py               # Package exports
│   ├── __main__.py               # CLI entry point (python -m ...)
│   ├── cli.py                    # Command-line interface
│   ├── model.py                  # TRM architecture implementation
│   ├── config.py                 # Configuration dataclasses
│   ├── trainer.py                # Training loop with deep supervision
│   ├── demo.py                   # Interactive demonstration
│   ├── data/                     # Dataset implementations
│   │   ├── __init__.py          # Dataset exports
│   │   ├── base.py              # Abstract base class
│   │   ├── sudoku.py            # Sudoku-Extreme dataset
│   │   ├── maze.py              # Maze-Hard dataset
│   │   ├── arc.py               # ARC-AGI dataset
│   │   └── arithmetic.py        # Simple arithmetic dataset
│   ├── tasks/                    # Task-specific utilities
│   │   └── __init__.py          # Task registry
│   └── utils/                    # Helper functions
│       └── __init__.py          # Seeding, device detection, etc.
│
├── train_maze.py                 # Standalone maze training script
├── train_arc.py                  # Standalone ARC training script
├── train_arithmetic.py           # Standalone arithmetic training script
│
├── tests/                        # Unit tests
│   ├── __init__.py
│   └── test_trm.py              # Model and training tests
│
├── examples/                     # Example scripts
│   └── custom_task_template.py  # Template for new tasks
│
├── pyproject.toml               # Modern Python packaging
├── setup.py                     # Legacy installation support
├── Makefile                     # Development commands
├── LICENSE                      # MIT license
├── CONTRIBUTING.md              # Contribution guidelines
└── README.md                    # This file
```

---

## Configuration

### Model Configuration (`TRMConfig`)

```python
from tiny_recursive_model import TRMConfig

config = TRMConfig(
    # Core architecture
    vocab_size=10,           # Token vocabulary size
    hidden_dim=256,          # Hidden dimension (d_model)
    num_layers=2,            # Transformer/MLP layers (paper uses 2)
    num_heads=8,             # Attention heads (if use_attention=True)
    context_length=81,       # Sequence length (81 for 9×9 Sudoku)
    
    # Recursion settings
    n_recursions=6,          # Latent updates per T step (paper: 4-6)
    T_steps=3,               # Answer updates per supervision step (paper: 3)
    n_supervision=16,        # Total supervision steps (paper: 16)
    
    # Architecture variant
    use_attention=False,     # False=MLP-Mixer, True=Self-Attention
    mlp_ratio=4.0,           # MLP expansion ratio
    
    # Regularization
    dropout=0.0,             # Dropout rate (paper uses 0)
)
```

### Training Configuration (`TrainingConfig`)

```python
from tiny_recursive_model import TrainingConfig

config = TrainingConfig(
    # Basic training
    batch_size=32,           # Batch size
    num_epochs=50,           # Number of epochs
    learning_rate=1e-4,      # Learning rate (paper: 1e-4)
    
    # Optimization
    weight_decay=0.1,        # AdamW weight decay (paper: 0.1-1.0)
    warmup_steps=500,        # LR warmup steps
    grad_clip=1.0,           # Gradient clipping
    
    # Deep supervision
    n_supervision=16,        # Must match model config
    n_supervision_eval=16,   # Supervision steps during evaluation
    
    # EMA (important for stability)
    use_ema=True,            # Use exponential moving average
    ema_decay=0.999,         # EMA decay rate
    
    # Logging
    log_interval=100,        # Log every N steps
    eval_interval=500,       # Evaluate every N steps
    save_interval=1000,      # Save checkpoint every N steps
    output_dir="outputs",    # Output directory
)
```

### Task-Specific Recommendations

**Sudoku-Extreme (9×9 fixed grid):**
```python
model_config = TRMConfig(
    use_attention=False,     # MLP-Mixer is better for fixed grids
    hidden_dim=256,
    n_recursions=6,
    T_steps=3,
    n_supervision=16,
)
train_config = TrainingConfig(
    weight_decay=0.1,
    num_epochs=50,
)
```

**Maze-Hard (30×30 variable paths):**
```python
model_config = TRMConfig(
    use_attention=True,      # Attention better for larger/variable grids
    hidden_dim=256,
    n_recursions=4,
    T_steps=3,
    n_supervision=16,
)
train_config = TrainingConfig(
    weight_decay=1.0,        # Paper uses higher weight decay for Maze
    num_epochs=5000,
)
```

**ARC-AGI (variable grids up to 30×30):**
```python
model_config = TRMConfig(
    use_attention=True,
    hidden_dim=256,
    n_recursions=4,
    T_steps=3,
    n_supervision=16,
)
train_config = TrainingConfig(
    weight_decay=1.0,
    num_epochs=100000,       # Paper trains for very long
)
```

---

## Training Your Own Models

### Custom Dataset Template

```python
import torch
from tiny_recursive_model.data import BaseDataset

class MyReasoningDataset(BaseDataset):
    """Template for creating custom reasoning datasets."""
    
    VOCAB_SIZE = 10  # Define your vocabulary size
    
    def __init__(self, n_samples=1000, context_length=64, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        
        self.context_length = context_length
        self.samples = []
        
        for _ in range(n_samples):
            # Generate your input-output pairs
            inp = self._generate_input()
            tgt = self._solve(inp)
            self.samples.append((inp, tgt))
    
    def _generate_input(self):
        """Generate a puzzle/problem instance."""
        # Return numpy array or list of token IDs
        return [0] * self.context_length
    
    def _solve(self, inp):
        """Compute the solution for the input."""
        # Return numpy array or list of token IDs
        return inp  # Placeholder
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        inp, tgt = self.samples[idx]
        return (
            torch.tensor(inp, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long)
        )
    
    @staticmethod
    def get_vocab_size():
        return MyReasoningDataset.VOCAB_SIZE
    
    def get_context_length(self):
        return self.context_length
```

### Full Training Script Template

```python
#!/usr/bin/env python3
"""Train TRM on a custom task."""

import torch
from torch.utils.data import DataLoader, random_split

from tiny_recursive_model import (
    TinyRecursiveModel,
    TRMConfig,
    TrainingConfig,
    Trainer,
)
from tiny_recursive_model.utils import set_seed, get_device

# Your custom dataset
from my_dataset import MyReasoningDataset

def main():
    set_seed(42)
    device = get_device()
    
    # Create dataset
    dataset = MyReasoningDataset(n_samples=5000)
    
    # Split train/test
    n_test = len(dataset) // 10
    train_set, test_set = random_split(dataset, [len(dataset) - n_test, n_test])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)
    
    # Create model
    model_config = TRMConfig(
        vocab_size=dataset.get_vocab_size(),
        context_length=dataset.get_context_length(),
        hidden_dim=256,
        num_layers=2,
        n_recursions=6,
        T_steps=3,
        n_supervision=16,
        use_attention=True,
    )
    
    model = TinyRecursiveModel(model_config)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    train_config = TrainingConfig(
        num_epochs=50,
        learning_rate=1e-4,
        use_ema=True,
    )
    
    trainer = Trainer(model, train_config, device=device)
    results = trainer.train(train_loader, test_loader, model_config)
    
    print(f"Final accuracy: {results['final_metrics']['cell_accuracy']:.2%}")

if __name__ == "__main__":
    main()
```


### Performance Tips

1. **Always use GPU** - Training is 10-50× faster
2. **Use EMA** - Critical for stability, always set `use_ema=True`
3. **Match paper settings** - `n_supervision=16`, `T_steps=3`, `n_recursions=6`
4. **Heavy augmentation** - Paper uses 1000× for Sudoku
5. **Appropriate weight decay** - 0.1 for Sudoku, 1.0 for Maze/ARC
6. **MLP-Mixer for fixed grids** - Use `use_attention=False` for Sudoku
7. **Self-Attention for variable grids** - Use `use_attention=True` for Maze/ARC

---

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@misc{jolicoeurmartineau2025more,
    title={Less is More: Recursive Reasoning with Tiny Networks}, 
    author={Alexia Jolicoeur-Martineau},
    year={2025},
    eprint={2510.04871},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2510.04871}, 
}
```

Also consider citing the Hierarchical Reasoning Model which inspired TRM:

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
    title={Hierarchical Reasoning Model}, 
    author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu 
            and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
    year={2025},
    eprint={2506.21734},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    url={https://arxiv.org/abs/2506.21734}, 
}
```

### Related Resources

| Resource | Link |
|----------|------|
| Original Paper | [arXiv:2510.04871](https://arxiv.org/abs/2510.04871) |
| HRM Paper | [arXiv:2506.21734](https://arxiv.org/abs/2506.21734) |
| ARC-AGI Benchmark | [arcprize.org](https://arcprize.org/) |

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This is an **educational reimplementation** of the TRM architecture. The original work and official code are by Samsung SAIL Montreal.

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas where contributions would be helpful:
- Additional benchmark datasets
- Performance optimizations
- Visualization tools
- Documentation improvements
- Bug fixes

---

## Acknowledgments

- **Alexia Jolicoeur-Martineau** and **Samsung SAIL Montreal** for the TRM paper
- **ARC Prize Foundation** for the ARC-AGI benchmark
- **HRM Authors** (Guan Wang et al.) for the foundational hierarchical reasoning approach
- The open-source ML community

---

<p align="center">
  <b>Less is More</b><br>
  <i>You don't always need to crank up model size for a model to reason and solve hard problems.</i>
</p>
