#!/usr/bin/env python3
"""Validate TF SavedModel/TF.js export vs PyTorch checkpoint."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
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
)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Compare PyTorch vs TF SavedModel outputs')
    parser.add_argument('--config', required=True, help='Config YAML usado en entrenamiento')
    parser.add_argument('--model', required=True, help='Checkpoint PyTorch (.pt)')
    parser.add_argument('--saved-model', required=True, help='Ruta SavedModel exportada')
    parser.add_argument('--val_jsonl', required=True, help='JSONL para extraer muestras')
    parser.add_argument('--class-mapping', help='JSON con mapping clase→índice')
    parser.add_argument('--train-jsonl', help='JSONL para derivar mapping si no existe archivo')
    parser.add_argument('--num-samples', type=int, default=128, help='Muestras a evaluar (default: 128)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size para inferencias')
    parser.add_argument('--device', help='cuda/cpu (auto por defecto)')
    parser.add_argument('--output', help='Archivo JSON para dejar métricas')
    return parser.parse_args()


def collect_samples(dataloader, max_samples):
    tensors = []
    labels = []
    for images, targets, _ in dataloader:
        tensors.append(images)
        labels.append(targets)
        if sum(t.shape[0] for t in tensors) >= max_samples:
            break

    stacked = torch.cat(tensors, dim=0)[:max_samples]
    labels = torch.cat(labels, dim=0)[:max_samples]
    return stacked, labels


def load_tf_signature(saved_model_path: Path):
    model = tf.saved_model.load(str(saved_model_path))
    if hasattr(model, 'signatures') and model.signatures:
        infer = next(iter(model.signatures.values()))
    else:
        infer = model
    input_keys = list(infer.structured_input_signature[1].keys())
    output_keys = list(infer.structured_outputs.keys())
    if not input_keys or not output_keys:
        raise ValueError('No se pudo inferir signature del SavedModel')
    return infer, input_keys[0], output_keys[0]


def main():
    args = parse_args()

    class_mapping = load_class_mapping(args.class_mapping, args.train_jsonl)
    dataloader, _, config = build_eval_dataloader(
        config_path=args.config,
        jsonl_path=args.val_jsonl,
        class_mapping=class_mapping,
        batch_size=args.batch_size,
        mode='val'
    )

    samples_tensor, labels = collect_samples(dataloader, args.num_samples)
    logger.info(f'Samples recogidos: {samples_tensor.shape[0]}')

    device = resolve_device(args.device)
    torch_model = load_model_from_checkpoint(config, args.model, len(class_mapping), device)

    with torch.no_grad():
        pytorch_logits = torch_model(samples_tensor.to(device)).cpu().numpy()

    tf_infer, input_key, output_key = load_tf_signature(Path(args.saved_model))
    tf_inputs = samples_tensor.numpy()
    tf_logits = []
    for start in tqdm(range(0, tf_inputs.shape[0], args.batch_size), desc='TF infer'):
        batch = tf_inputs[start:start + args.batch_size]
        outputs = tf_infer(**{input_key: tf.convert_to_tensor(batch)})
        tf_logits.append(outputs[output_key].numpy())
    tf_logits = np.vstack(tf_logits)

    diff = np.abs(pytorch_logits - tf_logits)
    mae = float(diff.mean())
    max_err = float(diff.max())

    torch_preds = pytorch_logits.argmax(axis=1)
    tf_preds = tf_logits.argmax(axis=1)
    agreement = float((torch_preds == tf_preds).mean())

    results = {
        'samples': int(samples_tensor.shape[0]),
        'mae': mae,
        'max_abs_diff': max_err,
        'top1_agreement': agreement,
        'pytorch_model': args.model,
        'saved_model': args.saved_model,
        'input_key': input_key,
        'output_key': output_key
    }

    logger.info('PyTorch vs TF - MAE %.6f | max diff %.6f | agreement %.4f', mae, max_err, agreement)

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
    else:
        print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
