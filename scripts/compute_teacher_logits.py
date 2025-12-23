#!/usr/bin/env python3
"""Genera logits de un teacher para distillation."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
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
    parser = argparse.ArgumentParser(description='Compute logits from a teacher model')
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--train-jsonl', required=True, help='JSONL con los samples para generar logits')
    parser.add_argument('--output', required=True, help='Archivo .npz (se usa np.savez_compressed)')
    parser.add_argument('--class-mapping')
    parser.add_argument('--save-class-mapping')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--device')
    parser.add_argument('--limit', type=int, help='Opcional: limitar número de muestras procesadas')
    return parser.parse_args()


def main():
    args = parse_args()

    class_mapping = load_class_mapping(args.class_mapping, args.train_jsonl)
    save_class_mapping(class_mapping, args.save_class_mapping)

    dataloader, dataset, config = build_eval_dataloader(
        config_path=args.config,
        jsonl_path=args.train_jsonl,
        class_mapping=class_mapping,
        batch_size=args.batch_size,
        mode='train'
    )

    device = resolve_device(args.device)
    model = load_model_from_checkpoint(config, args.model, len(class_mapping), device)

    logits_list = []
    labels_list = []
    latin_names = []
    plant_ids = []
    regions = []
    urls = []

    processed = 0

    with torch.no_grad():
        for images, labels, metadata in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            logits_list.append(outputs.detach().cpu().numpy())
            labels_list.append(labels.cpu().numpy())

            meta_batch = metadata
            if not isinstance(metadata, list):
                # default_collate devuelve dict de listas
                meta_batch = []
                if isinstance(metadata, dict):
                    batch_size = labels.size(0)
                    for idx in range(batch_size):
                        sample = {key: (metadata.get(key, [''] * batch_size)[idx] if isinstance(metadata.get(key), list) else metadata.get(key, '')) for key in metadata}
                        meta_batch.append(sample)
                else:
                    meta_batch = [{} for _ in range(labels.size(0))]

            for sample in meta_batch:
                latin_names.append(sample.get('latin_name', ''))
                plant_ids.append(sample.get('plant_id', ''))
                regions.append(sample.get('region', ''))
                urls.append(sample.get('url', ''))

            processed += labels.size(0)
            if args.limit and processed >= args.limit:
                logger.info(f"Limit alcanzado ({args.limit}). Deteniendo generación de logits.")
                break

    logits = np.concatenate(logits_list, axis=0).astype(np.float32)
    labels_arr = np.concatenate(labels_list, axis=0).astype(np.int64)

    if args.limit:
        logits = logits[:args.limit]
        labels_arr = labels_arr[:args.limit]
        latin_names[:] = latin_names[:args.limit]
        plant_ids[:] = plant_ids[:args.limit]
        regions[:] = regions[:args.limit]
        urls[:] = urls[:args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        logits=logits,
        labels=labels_arr,
        latin_names=np.array(latin_names, dtype=object),
        plant_ids=np.array(plant_ids, dtype=object),
        regions=np.array(regions, dtype=object),
        urls=np.array(urls, dtype=object),
        class_mapping_json=json.dumps(class_mapping)
    )
    logger.info(f"Logits guardados en {output_path} ({logits.shape[0]} muestras)")


if __name__ == '__main__':
    main()
