"""
Training Module
===============

Training logic for TRM with deep supervision, EMA, and ACT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import copy
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import asdict

from .config import TRMConfig, TrainingConfig, get_preset_config
from .model import TinyRecursiveModel
from .data import SudokuDataset, SimpleSudokuDataset
from .utils import (
    get_device, set_seed, count_parameters,
    save_checkpoint, load_checkpoint,
    EMAModel, StableMaxCrossEntropy,
    accuracy_score, puzzle_accuracy
)


class Trainer:
    """Trainer for Tiny Recursive Model.
    
    Implements the deep supervision training loop with:
    - Multiple supervision steps per sample
    - State carrying across steps
    - ACT (Adaptive Computation Time) for early stopping
    - EMA for weight averaging
    - Gradient clipping and warmup
    
    Parameters
    ----------
    model : TinyRecursiveModel
        Model to train
    train_config : TrainingConfig
        Training configuration
    device : torch.device, optional
        Device to use
    
    Examples
    --------
    >>> model = TinyRecursiveModel(model_config)
    >>> trainer = Trainer(model, train_config)
    >>> trainer.train(train_loader, test_loader)
    """
    
    def __init__(
        self,
        model: TinyRecursiveModel,
        train_config: TrainingConfig,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.config = train_config
        self.device = device or get_device()
        
        self.model = self.model.to(self.device)
        
        # EMA
        self.ema = None
        if train_config.use_ema:
            self.ema = EMAModel(model, decay=train_config.ema_decay)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Will be set when train() is called
        self.scheduler = None
        
        # Loss function
        self.criterion = StableMaxCrossEntropy()
        
        # Tracking
        self.global_step = 0
        self.best_metric = 0.0
        
        # Output directory
        self.output_dir = Path(train_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_scheduler(self, total_steps: int):
        """Create learning rate scheduler with warmup."""
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.config.warmup_steps
        )
        decay_steps = max(1, total_steps - self.config.warmup_steps)
        decay_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=decay_steps
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[self.config.warmup_steps]
        )
    
    def train_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, float]:
        """Perform one training step with deep supervision.
        
        Parameters
        ----------
        batch : Tuple[Tensor, Tensor]
            (input_tokens, target_tokens)
        
        Returns
        -------
        Dict[str, float]
            Training metrics for this step
        """
        self.model.train()
        
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Initialize states
        y, z = None, None
        
        total_loss = 0.0
        total_ce_loss = 0.0
        total_halt_loss = 0.0
        n_steps = 0
        
        # Deep supervision loop
        for step in range(self.config.n_supervision):
            self.optimizer.zero_grad()
            
            # Forward pass
            (y, z), logits, q_hat = self.model(inputs, y, z)
            
            # Cross-entropy loss
            ce_loss = self.criterion(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
            loss = ce_loss
            total_ce_loss += ce_loss.item()
            
            # Halting loss (ACT)
            if self.config.use_act:
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    is_correct = (preds == targets).all(dim=-1).float()
                
                halt_loss = F.binary_cross_entropy_with_logits(q_hat, is_correct)
                loss = loss + self.config.halt_loss_weight * halt_loss
                total_halt_loss += halt_loss.item()
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.grad_clip
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_steps += 1
            
            # Early stopping based on ACT
            if self.config.use_act:
                with torch.no_grad():
                    halt_prob = torch.sigmoid(q_hat).mean().item()
                    if halt_prob > 0.5:
                        break
        
        return {
            "loss": total_loss / n_steps,
            "ce_loss": total_ce_loss / n_steps,
            "halt_loss": total_halt_loss / n_steps if self.config.use_act else 0.0,
            "n_steps": n_steps
        }
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None
    ) -> Dict[str, float]:
        """Evaluate model on a dataset.
        
        Parameters
        ----------
        dataloader : DataLoader
            Evaluation data
        max_batches : int, optional
            Maximum batches to evaluate
        
        Returns
        -------
        Dict[str, float]
            Evaluation metrics
        """
        self.model.eval()
        
        all_preds = []
        all_targets = []
        total_loss = 0.0
        n_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Run full supervision steps
            y, z = None, None
            for _ in range(self.config.n_supervision_eval):
                (y, z), logits, _ = self.model(inputs, y, z)
            
            # Loss
            loss = self.criterion(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
            total_loss += loss.item()
            
            # Predictions
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            
            n_batches += 1
        
        # Aggregate
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        cell_acc = accuracy_score(all_preds, all_targets)
        puzz_acc = puzzle_accuracy(all_preds, all_targets)
        
        return {
            "loss": total_loss / n_batches,
            "cell_accuracy": cell_acc,
            "puzzle_accuracy": puzz_acc,
            "n_samples": len(all_preds)
        }
    
    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model_config: Optional[TRMConfig] = None
    ) -> Dict[str, Any]:
        """Run the full training loop.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data
        test_loader : DataLoader
            Test data
        model_config : TRMConfig, optional
            Model config (for saving)
        
        Returns
        -------
        Dict[str, Any]
            Training history and final metrics
        """
        total_steps = len(train_loader) * self.config.num_epochs
        self._create_scheduler(total_steps)
        
        # Save config
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump({
                "model_config": asdict(model_config) if model_config else {},
                "train_config": asdict(self.config)
            }, f, indent=2)
        
        print(f"Training for {self.config.num_epochs} epochs...")
        print(f"Total steps: {total_steps}")
        print(f"Device: {self.device}")
        print(f"Parameters: {count_parameters(self.model):,}")
        
        history = {"train_loss": [], "eval_metrics": []}
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            epoch_start = time.time()
            
            for batch_idx, batch in enumerate(train_loader):
                # Train step
                metrics = self.train_step(batch)
                
                # Update EMA
                if self.ema is not None:
                    self.ema.update(self.model)
                
                # Update scheduler
                self.scheduler.step()
                
                epoch_loss += metrics["loss"]
                epoch_steps += 1
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.log_interval == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    print(
                        f"Epoch {epoch+1}/{self.config.num_epochs} | "
                        f"Step {self.global_step} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"LR: {lr:.2e}"
                    )
                
                # Evaluation
                if self.global_step % self.config.eval_interval == 0:
                    # Use EMA weights for eval
                    if self.ema is not None:
                        self.ema.apply_shadow(self.model)
                    
                    eval_metrics = self.evaluate(test_loader, max_batches=25)
                    history["eval_metrics"].append({
                        "step": self.global_step,
                        **eval_metrics
                    })
                    
                    if self.ema is not None:
                        self.ema.restore(self.model)
                    
                    print(
                        f"  Eval | Loss: {eval_metrics['loss']:.4f} | "
                        f"Cell: {eval_metrics['cell_accuracy']:.1%} | "
                        f"Puzzle: {eval_metrics['puzzle_accuracy']:.1%}"
                    )
                    
                    # Save best
                    if eval_metrics['puzzle_accuracy'] > self.best_metric:
                        self.best_metric = eval_metrics['puzzle_accuracy']
                        self._save_best(model_config)
                
                # Periodic save
                if self.global_step % self.config.save_interval == 0:
                    save_checkpoint(
                        self.ema.shadow if self.ema else self.model,
                        self.optimizer,
                        self.scheduler,
                        epoch=epoch,
                        global_step=self.global_step,
                        path=self.output_dir / f"checkpoint_step{self.global_step}.pt"
                    )
            
            # End of epoch
            avg_loss = epoch_loss / epoch_steps
            epoch_time = time.time() - epoch_start
            history["train_loss"].append(avg_loss)
            print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")
        
        # Final evaluation
        print("\nFinal Evaluation...")
        if self.ema is not None:
            self.ema.apply_shadow(self.model)
        
        final_metrics = self.evaluate(test_loader)
        print(f"Final Cell Accuracy: {final_metrics['cell_accuracy']:.2%}")
        print(f"Final Puzzle Accuracy: {final_metrics['puzzle_accuracy']:.2%}")
        print(f"Best Puzzle Accuracy: {self.best_metric:.2%}")
        
        # Save final
        save_checkpoint(
            self.model,
            config=model_config,
            metrics=final_metrics,
            path=self.output_dir / "model_final.pt"
        )
        
        return {
            "history": history,
            "final_metrics": final_metrics,
            "best_metric": self.best_metric
        }
    
    def _save_best(self, model_config: Optional[TRMConfig] = None):
        """Save best model checkpoint."""
        save_checkpoint(
            self.ema.shadow if self.ema else self.model,
            config=model_config,
            metrics={"puzzle_accuracy": self.best_metric},
            global_step=self.global_step,
            path=self.output_dir / "model_best.pt"
        )


def train(
    task: str = "sudoku",
    model_config: Optional[TRMConfig] = None,
    train_config: Optional[TrainingConfig] = None,
    **kwargs
) -> Dict[str, Any]:
    """High-level training function.
    
    Parameters
    ----------
    task : str
        Task name ('sudoku', 'maze', 'arc')
    model_config : TRMConfig, optional
        Model configuration (uses preset if None)
    train_config : TrainingConfig, optional
        Training configuration (uses defaults if None)
    **kwargs
        Override training config values
    
    Returns
    -------
    Dict[str, Any]
        Training results
    
    Examples
    --------
    >>> results = train(task='sudoku', epochs=50, batch_size=32)
    """
    # Get configs
    if model_config is None:
        model_config = get_preset_config(task)
    
    if train_config is None:
        train_config = TrainingConfig()
    
    # Apply kwargs overrides
    for key, value in kwargs.items():
        if hasattr(train_config, key):
            setattr(train_config, key, value)
    
    # Set seed
    set_seed(train_config.seed)
    
    # Create datasets
    if task == "sudoku":
        hint_ranges = {"easy": (30, 35), "medium": (25, 30), "hard": (17, 24)}
        n_hints_range = hint_ranges.get(kwargs.get("difficulty", "medium"), (25, 30))
        
        train_dataset = SudokuDataset(
            n_puzzles=train_config.n_train_samples,
            n_hints_range=n_hints_range,
            augment=True,
            n_augmentations=train_config.n_augmentations,
            seed=train_config.seed
        )
        test_dataset = SimpleSudokuDataset(
            n_samples=train_config.n_test_samples,
            difficulty=kwargs.get("difficulty", "medium")
        )
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config.batch_size,
        shuffle=False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    model = TinyRecursiveModel(model_config)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Create trainer and train
    trainer = Trainer(model, train_config)
    results = trainer.train(train_loader, test_loader, model_config)
    
    return results
