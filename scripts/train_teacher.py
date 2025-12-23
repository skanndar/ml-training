#!/usr/bin/env python3
"""
train_teacher.py - Train Teacher Models (ViT-Base)

Trains teacher models on plant recognition dataset:
- Teacher Global: Full dataset (all regions)
- Teacher Regional: Regional subset (EU_SW)

Features:
- Streaming dataset with LRU cache
- Mixed precision training (FP16)
- Gradient accumulation
- Learning rate warmup + cosine decay
- Checkpoint saving/resuming
- TensorBoard logging
- Top-1/Top-5 accuracy metrics

Usage:
    # Debug mode (small subset)
    python scripts/train_teacher.py --config ./config/teacher_global.yaml --debug --max-samples 1000

    # Full training
    python scripts/train_teacher.py --config ./config/teacher_global.yaml

    # Resume from checkpoint
    python scripts/train_teacher.py --config ./config/teacher_global.yaml --resume ./checkpoints/teacher_global/last_checkpoint.pt
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.dataloader_factory import create_dataloaders

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsTracker:
    """Track training metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.loss_sum = 0.0
        self.top1_correct = 0
        self.top5_correct = 0
        self.total = 0
        self.predictions = []
        self.targets = []

    def update(self, loss, outputs, targets):
        """
        Update metrics with batch results.

        Args:
            loss: Batch loss (scalar, already averaged by criterion)
            outputs: Model outputs [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        """
        batch_size = targets.size(0)

        # Loss (already batch-averaged by CrossEntropyLoss)
        self.loss_sum += loss
        self.total += 1

        # Top-K accuracy
        _, pred = outputs.topk(5, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        # Top-1
        self.top1_correct += correct[:1].reshape(-1).float().sum(0).item()

        # Top-5
        self.top5_correct += correct[:5].reshape(-1).float().sum(0).item()

        # Store for confusion matrix
        self.predictions.extend(pred[0].cpu().numpy())
        self.targets.extend(targets.cpu().numpy())

    def get_metrics(self):
        """Get current metrics."""
        if self.total == 0:
            return {
                'loss': 0.0,
                'top1_acc': 0.0,
                'top5_acc': 0.0
            }

        # Note: self.total now tracks number of batches
        num_samples = len(self.targets)

        return {
            'loss': self.loss_sum / self.total,  # Average loss across batches
            'top1_acc': self.top1_correct / num_samples * 100,  # Accuracy over all samples
            'top5_acc': self.top5_correct / num_samples * 100   # Accuracy over all samples
        }


class Trainer:
    """Teacher model trainer."""

    def __init__(self, config_path: str, debug: bool = False, max_samples: int = None):
        """
        Args:
            config_path: Path to config YAML
            debug: Debug mode (small subset)
            max_samples: Max samples to use (for debugging)
        """
        self.config_path = config_path
        self.debug = debug
        self.max_samples = max_samples

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup device
        self.device = torch.device(
            self.config['device']['type'] if torch.cuda.is_available() else 'cpu'
        )
        logger.info(f"Using device: {self.device}")

        # Create output directories
        self.checkpoint_dir = Path(self.config['output']['checkpoint_dir'])
        self.results_dir = Path(self.config['output']['results_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.checkpoint_dir / 'logs'))

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_metric = 0.0

        # Mixed precision
        self.use_amp = self.config['device'].get('mixed_precision', False)
        self.scaler = GradScaler() if self.use_amp else None

        logger.info(f"Mixed precision training: {self.use_amp}")

    def create_model(self, num_classes: int):
        """Create model."""
        import timm

        model_name = self.config['model']['name']
        pretrained = self.config['model'].get('pretrained', True)

        logger.info(f"Creating model: {model_name}")
        logger.info(f"Pretrained: {pretrained}")
        logger.info(f"Num classes: {num_classes}")

        # Create model
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )

        # Load from checkpoint if specified
        init_from = self.config['model'].get('init_from', None)
        if init_from and Path(init_from).exists():
            logger.info(f"Loading weights from: {init_from}")
            checkpoint = torch.load(init_from, map_location='cpu')
            state_dict = checkpoint.get('model_state_dict', checkpoint)

            # Remove classifier heads if shapes mismatch (e.g., regional model has fewer clases)
            classifier_keys = [
                'head.weight', 'head.bias',
                'fc.weight', 'fc.bias',
                'classifier.weight', 'classifier.bias'
            ]
            model_state = model.state_dict()
            for key in classifier_keys:
                if key in state_dict and key in model_state:
                    if state_dict[key].shape != model_state[key].shape:
                        logger.warning(
                            f"Skipping pretrained layer '{key}' due to shape mismatch: "
                            f"{state_dict[key].shape} vs {model_state[key].shape}"
                        )
                        state_dict.pop(key)

            model.load_state_dict(state_dict, strict=False)

        model = model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        return model

    def create_optimizer(self, model):
        """Create optimizer."""
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training'].get('weight_decay', 0.01)

        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        logger.info(f"Optimizer: AdamW (lr={lr}, weight_decay={weight_decay})")

        return optimizer

    def create_scheduler(self, optimizer, num_training_steps: int):
        """Create learning rate scheduler."""
        warmup_epochs = self.config['training'].get('warmup_epochs', 0)
        total_epochs = self.config['training']['epochs']

        # Cosine annealing after warmup
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=1e-7
        )

        return scheduler, warmup_epochs

    def create_criterion(self):
        """Create loss function."""
        label_smoothing = self.config['training'].get('label_smoothing', 0.0)

        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        logger.info(f"Loss: CrossEntropyLoss (label_smoothing={label_smoothing})")

        return criterion

    def train_epoch(self, model, train_loader, optimizer, criterion, epoch):
        """Train one epoch."""
        model.train()
        metrics = MetricsTracker()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")

        for batch_idx, (images, labels, metadata) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                # Gradient clipping
                if self.config['training'].get('gradient_clip', 0) > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config['training'].get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                optimizer.step()

            # Update metrics
            metrics.update(loss.item(), outputs.detach(), labels)

            # Update progress bar
            current_metrics = metrics.get_metrics()
            pbar.set_postfix({
                'loss': f"{current_metrics['loss']:.4f}",
                'top1': f"{current_metrics['top1_acc']:.2f}%",
                'top5': f"{current_metrics['top5_acc']:.2f}%"
            })

            # Log to TensorBoard
            if batch_idx % self.config['logging'].get('log_every_n_steps', 100) == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], self.global_step)

            self.global_step += 1

        return metrics.get_metrics()

    @torch.no_grad()
    def validate(self, model, val_loader, criterion, epoch):
        """Validate model."""
        model.eval()
        metrics = MetricsTracker()

        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")

        for images, labels, metadata in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Update metrics
            metrics.update(loss.item(), outputs, labels)

            # Update progress bar
            current_metrics = metrics.get_metrics()
            pbar.set_postfix({
                'loss': f"{current_metrics['loss']:.4f}",
                'top1': f"{current_metrics['top1_acc']:.2f}%",
                'top5': f"{current_metrics['top5_acc']:.2f}%"
            })

        return metrics.get_metrics()

    def save_checkpoint(self, model, optimizer, scheduler, metrics, is_best=False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'best_val_metric': self.best_val_metric
        }

        # Save last checkpoint
        last_path = self.checkpoint_dir / 'last_checkpoint.pt'
        torch.save(checkpoint, last_path)
        logger.info(f"Saved checkpoint: {last_path}")

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"✅ New best model! Val Top-1: {metrics['val_top1_acc']:.2f}%")

        # Save epoch checkpoint
        if self.config['callbacks'].get('save_checkpoint_every', 0) > 0:
            if (self.epoch + 1) % self.config['callbacks']['save_checkpoint_every'] == 0:
                epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{self.epoch+1}.pt'
                torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, checkpoint_path: str, model, optimizer, scheduler):
        """Load checkpoint."""
        logger.info(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_metric = checkpoint.get('best_val_metric', 0.0)

        logger.info(f"Resumed from epoch {self.epoch}, step {self.global_step}")

    def train(self, resume_from: str = None):
        """Main training loop."""
        logger.info("="*70)
        logger.info("Starting Teacher Training")
        logger.info("="*70)

        # Create dataloaders
        logger.info("\nCreating dataloaders...")
        train_loader, val_loader, info = create_dataloaders(self.config_path)

        # Debug mode: limit samples
        if self.debug and self.max_samples:
            logger.info(f"\n🐛 DEBUG MODE: Using max {self.max_samples} samples")
            train_loader.dataset.records = train_loader.dataset.records[:self.max_samples]
            val_loader.dataset.records = val_loader.dataset.records[:min(self.max_samples//10, len(val_loader.dataset.records))]
            logger.info(f"Train: {len(train_loader.dataset)} samples")
            logger.info(f"Val: {len(val_loader.dataset)} samples")

        num_classes = info['num_classes']

        # Create model
        logger.info("\nCreating model...")
        model = self.create_model(num_classes)

        # Create optimizer and scheduler
        optimizer = self.create_optimizer(model)
        scheduler, warmup_epochs = self.create_scheduler(optimizer, len(train_loader))

        # Create loss
        criterion = self.create_criterion()

        # Resume from checkpoint
        if resume_from and Path(resume_from).exists():
            self.load_checkpoint(resume_from, model, optimizer, scheduler)

        # Training loop
        logger.info("\n" + "="*70)
        logger.info("Starting training loop")
        logger.info("="*70)

        num_epochs = self.config['training']['epochs']
        history = {
            'train_loss': [],
            'train_top1_acc': [],
            'train_top5_acc': [],
            'val_loss': [],
            'val_top1_acc': [],
            'val_top5_acc': []
        }

        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch

            # Warmup learning rate
            if epoch < warmup_epochs:
                lr_scale = (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.config['training']['learning_rate'] * lr_scale

            # Train
            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion, epoch)

            # Validate
            val_metrics = self.validate(model, val_loader, criterion, epoch)

            # Step scheduler
            if epoch >= warmup_epochs:
                scheduler.step()

            # Log metrics
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, Top-1: {train_metrics['top1_acc']:.2f}%, Top-5: {train_metrics['top5_acc']:.2f}%")
            logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, Top-1: {val_metrics['top1_acc']:.2f}%, Top-5: {val_metrics['top5_acc']:.2f}%")

            # TensorBoard
            self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/train_top1_acc', train_metrics['top1_acc'], epoch)
            self.writer.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/val_top1_acc', val_metrics['top1_acc'], epoch)

            # Save history
            history['train_loss'].append(train_metrics['loss'])
            history['train_top1_acc'].append(train_metrics['top1_acc'])
            history['train_top5_acc'].append(train_metrics['top5_acc'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_top1_acc'].append(val_metrics['top1_acc'])
            history['val_top5_acc'].append(val_metrics['top5_acc'])

            # Save checkpoint
            checkpoint_metric = self.config['callbacks'].get('checkpoint_metric', 'top1_acc')
            is_best = val_metrics[checkpoint_metric] > self.best_val_metric

            if is_best:
                self.best_val_metric = val_metrics[checkpoint_metric]

            combined_metrics = {**{f'train_{k}': v for k, v in train_metrics.items()},
                              **{f'val_{k}': v for k, v in val_metrics.items()}}

            self.save_checkpoint(model, optimizer, scheduler, combined_metrics, is_best)

            # Early stopping
            if self.config['callbacks'].get('early_stopping', False):
                patience = self.config['callbacks'].get('early_stopping_patience', 5)
                # Implement early stopping logic here if needed

        # Save final results
        results_path = self.results_dir / 'training_history.json'
        try:
            with open(results_path, 'w') as f:
                json.dump(history, f, indent=2)
            logger.info(f"\n✅ Training complete! Results saved to: {results_path}")
            logger.info(f"Best validation {checkpoint_metric}: {self.best_val_metric:.2f}%")
        except Exception as e:
            logger.error(f"❌ Failed to save training history: {e}")
            raise

        # Close TensorBoard writer
        try:
            self.writer.close()
            logger.info("✅ TensorBoard writer closed successfully")
        except Exception as e:
            logger.warning(f"⚠️  Warning closing TensorBoard writer: {e}")
            # Don't raise - this is non-critical


def main():
    parser = argparse.ArgumentParser(description='Train teacher model')
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Path to config YAML')
    parser.add_argument('--resume', '-r', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode (small subset)')
    parser.add_argument('--max-samples', type=int, default=1000,
                       help='Max samples in debug mode')

    args = parser.parse_args()

    # Create trainer
    trainer = Trainer(
        config_path=args.config,
        debug=args.debug,
        max_samples=args.max_samples if args.debug else None
    )

    # Train
    trainer.train(resume_from=args.resume)


if __name__ == '__main__':
    main()
