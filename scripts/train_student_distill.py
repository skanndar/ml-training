#!/usr/bin/env python3
"""Train student model via knowledge distillation."""

import argparse
import json
import logging
from pathlib import Path
import random

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.dataloader_factory import DataLoaderFactory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss_sum = 0.0
        self.top1_correct = 0.0
        self.top5_correct = 0.0
        self.total_batches = 0
        self.total_samples = 0

    def update(self, loss, outputs, targets):
        batch_size = targets.size(0)
        self.loss_sum += loss
        self.total_batches += 1
        self.total_samples += batch_size

        maxk = min(5, outputs.size(1))
        _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        self.top1_correct += correct[:1].reshape(-1).float().sum().item()
        self.top5_correct += correct[:maxk].reshape(-1).float().sum().item()

    def get_metrics(self):
        return {
            'loss': self.loss_sum / self.total_batches if self.total_batches else 0.0,
            'top1_acc': (self.top1_correct / self.total_samples * 100) if self.total_samples else 0.0,
            'top5_acc': (self.top5_correct / self.total_samples * 100) if self.total_samples else 0.0
        }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class StudentDistillationTrainer:
    def __init__(self, config_path: str, debug: bool = False, max_samples: int = None):
        self.config_path = Path(config_path)
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.debug = debug
        self.max_samples = max_samples if debug else None

        self.device = torch.device(self.config['device'].get('type', 'cuda') if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        self.checkpoint_dir = Path(self.config['output']['checkpoint_dir'])
        self.results_dir = Path(self.config['output']['results_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.checkpoint_dir / 'logs'))

        self.use_amp = self.config['device'].get('mixed_precision', False)
        self.scaler = GradScaler() if self.use_amp else None

        self.epoch = 0
        self.global_step = 0
        self.best_val_metric = 0.0

        self.class_mapping = None
        self.soft_label_map = {}
        self._load_soft_labels()

    def _load_soft_labels(self):
        soft_cfg = self.config['soft_labels']
        npz_path = Path(soft_cfg['path'])
        if not npz_path.exists():
            raise FileNotFoundError(f"Soft labels npz no encontrado: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)
        if 'soft_probs' not in data.files:
            raise ValueError("El archivo de soft labels debe contener 'soft_probs'")

        soft_probs = data['soft_probs']
        plant_ids = data['plant_ids']
        mapping_str = '{}'
        if 'class_mapping_json' in data.files:
            mapping_str = data['class_mapping_json'].item()
        elif self.config['soft_labels'].get('class_mapping_json'):
            mapping_path = Path(self.config['soft_labels']['class_mapping_json'])
            if mapping_path.exists():
                mapping_str = mapping_path.read_text(encoding='utf-8')
        self.class_mapping = {k: int(v) for k, v in json.loads(mapping_str).items()}
        if not self.class_mapping:
            raise ValueError('No se pudo cargar class_mapping para el student')

        for pid, probs in zip(plant_ids, soft_probs):
            self.soft_label_map[str(pid)] = probs

        logger.info(f"Soft labels cargados: {len(self.soft_label_map)} muestras, {soft_probs.shape[1]} clases")

    def _build_dataloaders(self):
        factory = DataLoaderFactory(self.config_path)

        train_dataset = factory.create_dataset(
            self.config['data']['train_jsonl'],
            mode='train',
            class_to_idx=self.class_mapping
        )
        val_dataset = factory.create_dataset(
            self.config['data']['val_jsonl'],
            mode='val',
            class_to_idx=self.class_mapping
        )

        train_loader = factory.create_dataloader(train_dataset, mode='train')
        val_loader = factory.create_dataloader(val_dataset, mode='val')
        info = {
            'num_classes': len(self.class_mapping),
            'train_size': len(train_dataset),
            'val_size': len(val_dataset)
        }
        return train_loader, val_loader, info

    def create_model(self, num_classes: int):
        model_cfg = self.config['model']
        model = timm.create_model(
            model_cfg['name'],
            pretrained=model_cfg.get('pretrained', True),
            num_classes=num_classes
        )
        model = model.to(self.device)
        return model

    def create_optimizer(self, model):
        optim_cfg = self.config['training']
        return AdamW(
            model.parameters(),
            lr=optim_cfg['learning_rate'],
            weight_decay=optim_cfg.get('weight_decay', 0.0)
        )

    def create_scheduler(self, optimizer, steps_per_epoch):
        epochs = self.config['training']['epochs']
        warmup = self.config['training'].get('warmup_epochs', 0)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - warmup)
        )
        return scheduler, warmup

    def create_criterion(self):
        smoothing = self.config['training'].get('label_smoothing', 0.0)
        return torch.nn.CrossEntropyLoss(label_smoothing=smoothing)

    def _get_soft_batch(self, metadata, labels, num_classes):
        plant_ids = metadata.get('plant_id', [])
        if isinstance(plant_ids, list):
            ids = plant_ids
        else:
            ids = list(plant_ids)

        batch_probs = []
        missing = 0
        for pid, label in zip(ids, labels):
            probs = self.soft_label_map.get(str(pid))
            if probs is None:
                vec = torch.zeros(num_classes, dtype=torch.float32)
                vec[label.item()] = 1.0
                missing += 1
            else:
                vec = torch.from_numpy(probs.astype(np.float32))
            batch_probs.append(vec)
        if missing:
            logger.debug(f"Soft labels faltantes en {missing} muestras")
        return torch.stack(batch_probs).to(self.device)

    def save_checkpoint(self, model, optimizer, scheduler, metrics, is_best=False):
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
            'best_val_metric': self.best_val_metric,
            'config': self.config
        }

        last_path = self.checkpoint_dir / 'last_checkpoint.pt'
        torch.save(checkpoint, last_path)
        logger.info(f"Saved checkpoint: {last_path}")

        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info("✅ Nuevo best checkpoint guardado")

    def train(self):
        train_loader, val_loader, info = self._build_dataloaders()

        if self.debug and self.max_samples:
            train_loader.dataset.records = train_loader.dataset.records[:self.max_samples]
            val_loader.dataset.records = val_loader.dataset.records[:min(self.max_samples // 10, len(val_loader.dataset.records))]

        num_classes = info['num_classes']
        model = self.create_model(num_classes)
        optimizer = self.create_optimizer(model)
        scheduler, warmup_epochs = self.create_scheduler(optimizer, len(train_loader))
        ce_loss = self.create_criterion()

        num_epochs = self.config['training']['epochs']
        alpha = self.config['loss']['alpha']
        beta = self.config['loss']['beta']
        temperature = self.config['loss']['temperature']
        grad_accum = max(1, self.config['training'].get('grad_accum_steps', 1))

        logger.info(f"Iniciando entrenamiento student: {num_epochs} epochs")

        history = []

        for epoch in range(num_epochs):
            self.epoch = epoch

            if epoch < warmup_epochs:
                lr_scale = (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.config['training']['learning_rate'] * lr_scale

            train_metrics = self.train_epoch(model, train_loader, optimizer, ce_loss, alpha, beta, temperature, num_classes, grad_accum)
            val_metrics = self.validate(model, val_loader, ce_loss)

            if epoch >= warmup_epochs:
                scheduler.step()

            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train loss {train_metrics['loss']:.4f}, Top1 {train_metrics['top1_acc']:.2f}% | Val loss {val_metrics['loss']:.4f}, Top1 {val_metrics['top1_acc']:.2f}%")

            self.writer.add_scalar('train/loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('train/top1', train_metrics['top1_acc'], epoch)
            self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('val/top1', val_metrics['top1_acc'], epoch)

            is_best = val_metrics['top1_acc'] > self.best_val_metric
            if is_best:
                self.best_val_metric = val_metrics['top1_acc']
            metrics_combined = {**{f'train_{k}': v for k, v in train_metrics.items()}, **{f'val_{k}': v for k, v in val_metrics.items()}}
            self.save_checkpoint(model, optimizer, scheduler, metrics_combined, is_best)

            history.append({
                'epoch': epoch + 1,
                'train': train_metrics,
                'val': val_metrics
            })

        history_path = self.results_dir / 'training_history.json'
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        self.writer.close()
        logger.info(f"Entrenamiento student finalizado. Historia guardada en {history_path}")

    def train_epoch(self, model, dataloader, optimizer, ce_loss, alpha, beta, temperature, num_classes, grad_accum):
        model.train()
        tracker = MetricsTracker()
        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch+1} [Train]")
        optimizer.zero_grad()

        for batch_idx, (images, labels, metadata) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            if self.use_amp:
                with autocast():
                    outputs = model(images)
                    soft_batch = self._get_soft_batch(metadata, labels, num_classes)
                    distill_loss = F.kl_div(
                        F.log_softmax(outputs / temperature, dim=1),
                        soft_batch,
                        reduction='batchmean'
                    ) * (temperature ** 2)
                    ce = ce_loss(outputs, labels)
                    loss = (alpha * distill_loss + beta * ce) / grad_accum
                self.scaler.scale(loss).backward()
            else:
                outputs = model(images)
                soft_batch = self._get_soft_batch(metadata, labels, num_classes)
                distill_loss = F.kl_div(
                    F.log_softmax(outputs / temperature, dim=1),
                    soft_batch,
                    reduction='batchmean'
                ) * (temperature ** 2)
                ce = ce_loss(outputs, labels)
                loss = (alpha * distill_loss + beta * ce) / grad_accum
                loss.backward()

            do_step = (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(dataloader)
            if do_step:
                if self.use_amp:
                    if self.config['training'].get('gradient_clip'):
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['training']['gradient_clip'])
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    if self.config['training'].get('gradient_clip'):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['training']['gradient_clip'])
                    optimizer.step()
                optimizer.zero_grad()

            tracker.update(loss.item() * grad_accum, outputs.detach(), labels)
            current = tracker.get_metrics()
            pbar.set_postfix({
                'loss': f"{current['loss']:.4f}",
                'top1': f"{current['top1_acc']:.2f}%",
                'top5': f"{current['top5_acc']:.2f}%"
            })
            self.global_step += 1

        return tracker.get_metrics()

    @torch.no_grad()
    def validate(self, model, dataloader, ce_loss):
        model.eval()
        tracker = MetricsTracker()
        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch+1} [Val]")
        for images, labels, metadata in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = model(images)
            loss = ce_loss(outputs, labels)
            tracker.update(loss.item(), outputs, labels)
            current = tracker.get_metrics()
            pbar.set_postfix({
                'loss': f"{current['loss']:.4f}",
                'top1': f"{current['top1_acc']:.2f}%",
                'top5': f"{current['top5_acc']:.2f}%"
            })
        return tracker.get_metrics()


def main():
    parser = argparse.ArgumentParser(description='Train student via distillation')
    parser.add_argument('--config', default='./config/student.yaml')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    trainer = StudentDistillationTrainer(args.config, args.debug, args.max_samples)
    trainer.train()


if __name__ == '__main__':
    main()
