#!/usr/bin/env python3
"""Evaluate a teacher checkpoint on un split específico."""

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml

from eval_utils import (
    build_eval_dataloader,
    load_class_mapping,
    load_model_from_checkpoint,
    resolve_device,
    save_class_mapping,
)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate teacher checkpoint')
    parser.add_argument('--config', required=True, help='Config YAML usado para el teacher (ej. config/teacher_regional.yaml)')
    parser.add_argument('--model', required=True, help='Ruta al checkpoint (.pt) a evaluar')
    parser.add_argument('--test-jsonl', required=True, help='JSONL con los samples de evaluación')
    parser.add_argument('--output', required=True, help='Archivo JSON para guardar resultados')
    parser.add_argument('--class-mapping', help='Archivo JSON con mapping clase→índice (opcional)')
    parser.add_argument('--train-jsonl', help='JSONL de train para reconstruir mapping si no existe el archivo')
    parser.add_argument('--save-class-mapping', help='Ruta para guardar el mapping derivado (opcional)')
    parser.add_argument('--batch-size', type=int, help='Batch size para la evaluación (default: config.training.batch_size)')
    parser.add_argument('--device', help='cuda o cpu (auto si no se especifica)')
    return parser.parse_args()


def compute_metrics(model, dataloader, device):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    total_batches = 0
    loss_sum = 0.0
    top1_correct = 0
    top5_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_sum += loss.item()
            total_batches += 1

            maxk = min(5, outputs.size(1))
            _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(labels.view(1, -1).expand_as(pred))

            top1_correct += correct[:1].reshape(-1).float().sum().item()
            top5_correct += correct[:maxk].reshape(-1).float().sum().item()
            total_samples += labels.size(0)

    avg_loss = loss_sum / total_batches if total_batches else 0.0
    top1 = (top1_correct / total_samples * 100) if total_samples else 0.0
    top5 = (top5_correct / total_samples * 100) if total_samples else 0.0

    return {
        'loss': avg_loss,
        'top1_accuracy': top1,
        'top5_accuracy': top5,
        'samples': total_samples
    }


def main():
    args = parse_args()

    class_mapping = load_class_mapping(args.class_mapping, args.train_jsonl)
    save_class_mapping(class_mapping, args.save_class_mapping)

    dataloader, dataset, config = build_eval_dataloader(
        config_path=args.config,
        jsonl_path=args.test_jsonl,
        class_mapping=class_mapping,
        batch_size=args.batch_size,
        mode='val'
    )

    device = resolve_device(args.device)
    model = load_model_from_checkpoint(config, args.model, len(class_mapping), device)

    metrics = compute_metrics(model, dataloader, device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'config': args.config,
        'checkpoint': args.model,
        'test_jsonl': args.test_jsonl,
        'class_mapping_size': len(class_mapping),
        **metrics
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Resultados guardados en {output_path}")
    logger.info(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
