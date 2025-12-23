#!/usr/bin/env python3
"""Evalúa un checkpoint calculando métricas por región."""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import torch

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
    parser = argparse.ArgumentParser(description='Eval teacher por región')
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--test-jsonl', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--class-mapping')
    parser.add_argument('--train-jsonl')
    parser.add_argument('--save-class-mapping')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--device')
    return parser.parse_args()


def to_list(metadata_entry, batch_size):
    if isinstance(metadata_entry, list):
        return metadata_entry
    if isinstance(metadata_entry, tuple):
        return list(metadata_entry)
    # dict of lists (default_collate)
    if isinstance(metadata_entry, dict):
        # Should not happen here
        return [metadata_entry.get(str(i), '') for i in range(batch_size)]
    # scalar -> replicate
    return [metadata_entry for _ in range(batch_size)]


def extract_field(metadata, key, batch_size):
    if isinstance(metadata, dict):
        value = metadata.get(key)
        if isinstance(value, list):
            return value
        if value is None:
            return [''] * batch_size
        return to_list(value, batch_size)
    if isinstance(metadata, list):
        return [item.get(key, '') for item in metadata]
    return [''] * batch_size


def evaluate_by_region(model, dataloader, device):
    criterion = torch.nn.CrossEntropyLoss()
    overall = {
        'loss_sum': 0.0,
        'batches': 0,
        'top1_correct': 0,
        'top5_correct': 0,
        'samples': 0
    }
    region_stats = defaultdict(lambda: {'count': 0, 'top1_correct': 0, 'top5_correct': 0})

    with torch.no_grad():
        for images, labels, metadata in dataloader:
            batch_size = labels.size(0)
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            overall['loss_sum'] += loss.item()
            overall['batches'] += 1
            overall['samples'] += batch_size

            maxk = min(5, outputs.size(1))
            _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
            pred = pred.cpu()
            labels_cpu = labels.cpu()

            regions = extract_field(metadata, 'region', batch_size)
            for idx in range(batch_size):
                region = regions[idx] or 'UNKNOWN'
                stats = region_stats[region]
                stats['count'] += 1

                topk = pred[idx]
                if topk[0].item() == labels_cpu[idx].item():
                    stats['top1_correct'] += 1
                    overall['top1_correct'] += 1
                else:
                    overall['top1_correct'] += 0

                if labels_cpu[idx].item() in topk[:maxk].tolist():
                    stats['top5_correct'] += 1
                    overall['top5_correct'] += 1
                else:
                    overall['top5_correct'] += 0

    results = {
        'loss': overall['loss_sum'] / overall['batches'] if overall['batches'] else 0.0,
        'top1_accuracy': (overall['top1_correct'] / overall['samples'] * 100) if overall['samples'] else 0.0,
        'top5_accuracy': (overall['top5_correct'] / overall['samples'] * 100) if overall['samples'] else 0.0,
        'samples': overall['samples']
    }

    per_region = {}
    for region, stats in region_stats.items():
        per_region[region] = {
            'samples': stats['count'],
            'top1_accuracy': stats['top1_correct'] / stats['count'] * 100 if stats['count'] else 0.0,
            'top5_accuracy': stats['top5_correct'] / stats['count'] * 100 if stats['count'] else 0.0
        }

    return results, per_region


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

    results, per_region = evaluate_by_region(model, dataloader, device)

    payload = {
        'config': args.config,
        'checkpoint': args.model,
        'test_jsonl': args.test_jsonl,
        'samples': results['samples'],
        'loss': results['loss'],
        'top1_accuracy': results['top1_accuracy'],
        'top5_accuracy': results['top5_accuracy'],
        'per_region': per_region
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Resultados por región guardados en {output_path}")


if __name__ == '__main__':
    main()
