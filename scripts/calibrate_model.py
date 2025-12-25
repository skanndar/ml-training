#!/usr/bin/env python3
"""Compute calibration metrics (ECE/MCE) for a checkpoint."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_utils import (  # type: ignore
    build_eval_dataloader,
    load_class_mapping,
    load_model_from_checkpoint,
    resolve_device,
    save_class_mapping,
)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Calibrate student/teacher checkpoint')
    parser.add_argument('--config', required=True, help='Config YAML (student/teacher)')
    parser.add_argument('--model', required=True, help='Checkpoint a evaluar (.pt)')
    parser.add_argument('--val_jsonl', required=True, help='JSONL con split de validación')
    parser.add_argument('--output', required=True, help='Archivo JSON para guardar resultados')
    parser.add_argument('--class-mapping', help='JSON con mapping clase→índice')
    parser.add_argument('--train-jsonl', help='JSONL para derivar mapping si no se proporciona archivo')
    parser.add_argument('--save-class-mapping', help='Ruta para guardar el mapping derivado (opcional)')
    parser.add_argument('--batch-size', type=int, help='Batch size de inferencia')
    parser.add_argument('--device', help='cuda/cpu (auto si no se especifica)')
    parser.add_argument('--bins', type=int, default=15, help='Número de bins para ECE/MCE (default: 15)')
    return parser.parse_args()


def compute_calibration_stats(confidences, correct, n_bins=15):
    confidences = np.asarray(confidences)
    correct = np.asarray(correct).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_stats = []

    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        count = mask.sum()
        if count == 0:
            bin_stats.append({
                'bin_start': float(bins[i]),
                'bin_end': float(bins[i + 1]),
                'accuracy': None,
                'confidence': None,
                'count': 0,
            })
            continue
        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        gap = abs(acc - conf)
        ece += (count / len(confidences)) * gap
        mce = max(mce, gap)
        bin_stats.append({
            'bin_start': float(bins[i]),
            'bin_end': float(bins[i + 1]),
            'accuracy': float(acc),
            'confidence': float(conf),
            'count': int(count),
        })

    return ece, mce, bin_stats


def main():
    args = parse_args()

    class_mapping = load_class_mapping(args.class_mapping, args.train_jsonl)
    save_class_mapping(class_mapping, args.save_class_mapping)

    dataloader, dataset, config = build_eval_dataloader(
        config_path=args.config,
        jsonl_path=args.val_jsonl,
        class_mapping=class_mapping,
        batch_size=args.batch_size,
        mode='val'
    )

    device = resolve_device(args.device)
    model = load_model_from_checkpoint(config, args.model, len(class_mapping), device)

    confidences = []
    correct = []
    predictions = []
    true_labels = []

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc='Calibrating'):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, dim=1)
            confidences.extend(conf.cpu().numpy().tolist())
            predictions.extend(pred.cpu().numpy().tolist())
            true_labels.extend(labels.cpu().numpy().tolist())
            correct.extend(pred.eq(labels).float().cpu().numpy().tolist())

    ece, mce, bin_stats = compute_calibration_stats(confidences, correct, args.bins)

    payload = {
        'config': args.config,
        'checkpoint': args.model,
        'val_jsonl': args.val_jsonl,
        'class_mapping_size': len(class_mapping),
        'samples': len(confidences),
        'ece': ece,
        'mce': mce,
        'bin_stats': bin_stats,
        'confidences': confidences,
        'correct': correct,
        'predictions': predictions,
        'labels': true_labels,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Calibración guardada en {output_path} (ECE={ece:.4f}, MCE={mce:.4f})")


if __name__ == '__main__':
    main()
