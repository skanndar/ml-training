#!/usr/bin/env python3
"""Combina logits de múltiples teachers en un único archivo de soft labels."""

import argparse
import json
import logging
from pathlib import Path
from typing import List

import numpy as np

# Metadata keys esperados en los npz
META_KEYS = ['latin_names', 'regions', 'urls']

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Combine teacher logits (.npz) into soft labels file')
    parser.add_argument('--teachers', nargs='+', required=True,
                        help='Lista de archivos .npz con logits (ej. teacher_global_logits_train.npz ...)')
    parser.add_argument('--weights', nargs='+', type=float,
                        help='Pesos (mismo número que teachers). Se normalizan automáticamente.')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperatura para el softmax final (default: 1.0)')
    parser.add_argument('--target-class-mapping', required=False,
                        help='JSON con mapping clase→índice objetivo (recomendado: data/class_mapping.json). Si se omite, se usa el mapping del primer teacher.')
    parser.add_argument('--output', required=True,
                        help='Archivo .npz de salida (soft labels combinados)')
    return parser.parse_args()


def softmax_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / temperature
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / np.sum(exp, axis=1, keepdims=True)


def main():
    args = parse_args()
    teacher_paths = [Path(p) for p in args.teachers]

    if args.weights:
        if len(args.weights) != len(teacher_paths):
            raise ValueError('Debes proveer el mismo número de pesos que de teachers')
        weights = np.array(args.weights, dtype=np.float64)
    else:
        weights = np.ones(len(teacher_paths), dtype=np.float64)

    weights = weights / weights.sum()
    logger.info(f'Pesos normalizados: {weights.tolist()}')

    teacher_data: List[dict] = []
    target_mapping = None
    if args.target_class_mapping:
        mapping_file = Path(args.target_class_mapping)
        if not mapping_file.exists():
            raise FileNotFoundError(f"No existe target-class-mapping: {mapping_file}")
        target_mapping = {str(k): int(v) for k, v in json.loads(mapping_file.read_text(encoding='utf-8')).items()}
        logger.info(f"Mapping objetivo cargado ({len(target_mapping)} clases)")

    for idx_teacher, path in enumerate(teacher_paths):
        if not path.exists():
            raise FileNotFoundError(f'No existe {path}')
        data = np.load(path, allow_pickle=True)
        if 'class_mapping_json' not in data.files:
            raise ValueError(f"El archivo {path} no contiene class_mapping_json")
        teacher_mapping = {str(k): int(v) for k, v in json.loads(data['class_mapping_json'].item()).items()}
        if target_mapping is None and idx_teacher == 0:
            target_mapping = dict(teacher_mapping)
            logger.info(f"Mapping objetivo derivado del primer teacher ({len(target_mapping)} clases)")
        teacher_entry = {
            'path': str(path),
            'logits': data['logits'],
            'labels': data['labels'],
            'metadata': {key: data[key] if key in data.files else None for key in META_KEYS},
            'plant_ids': data['plant_ids'] if 'plant_ids' in data.files else None,
            'mapping': teacher_mapping
        }
        teacher_data.append(teacher_entry)
        if teacher_entry['plant_ids'] is None:
            raise ValueError(f"El archivo {path} no contiene 'plant_ids'; es requerido para alinear logits")
    if target_mapping is None:
        raise ValueError('No se pudo determinar un mapping objetivo de clases')
    num_target_classes = len(target_mapping)
    logger.info(f"Clases objetivo totales: {num_target_classes}")
    # Unión de plant_ids, acumulando logits ponderados por los pesos disponibles
    logits_dim = num_target_classes
    accum_probs = {}
    accum_weights = {}
    labels_map = {}
    metadata_map = {key: {} for key in META_KEYS}
    class_mapping_json = json.dumps(target_mapping)

    for weight, entry in zip(weights, teacher_data):
        logger.info(f"Procesando {entry['path']} (peso {weight:.3f})")
        metadata_arrays = entry['metadata']
        teacher_logits = entry['logits']
        teacher_classes = list(entry['mapping'].keys())
        teacher_indices = np.array([entry['mapping'][cls] for cls in teacher_classes])
        target_indices = np.array([target_mapping[cls] for cls in teacher_classes])

        url_arr = entry['metadata'].get('urls')
        if url_arr is not None:
            keys = [str(u) for u in url_arr]
        else:
            keys = [str(pid) for pid in entry['plant_ids']]

        for idx, (sample_key, label_idx) in enumerate(zip(keys, entry['labels'])):
            logits_row = teacher_logits[idx, teacher_indices]
            vector = np.full(num_target_classes, -1e9, dtype=np.float32)
            vector[target_indices] = logits_row
            prob_row = softmax_logits(vector[None, :], 1.0)[0]
            if sample_key not in accum_probs:
                accum_probs[sample_key] = np.zeros(logits_dim, dtype=np.float32)
                accum_weights[sample_key] = 0.0
            accum_probs[sample_key] += weight * prob_row
            accum_weights[sample_key] += weight
            if sample_key not in labels_map:
                labels_map[sample_key] = label_idx
            for meta_key in META_KEYS:
                source_arr = metadata_arrays.get(meta_key)
                if source_arr is not None and sample_key not in metadata_map[meta_key]:
                    metadata_map[meta_key][sample_key] = source_arr[idx]

    sorted_ids = sorted(accum_probs.keys())
    logger.info(f"Total muestras combinadas: {len(sorted_ids)}")

    combined_labels = np.zeros(len(sorted_ids), dtype=np.int64)
    soft_probs = np.zeros((len(sorted_ids), logits_dim), dtype=np.float16)
    metadata_out = {key: np.empty(len(sorted_ids), dtype=object) for key in META_KEYS}
    for arr in metadata_out.values():
        arr[:] = ''

    for idx, sample_key in enumerate(sorted_ids):
        vector = accum_probs[sample_key] / max(accum_weights[sample_key], 1e-8)
        combined_labels[idx] = labels_map[sample_key]
        for meta_key in META_KEYS:
            if sample_key in metadata_map[meta_key]:
                metadata_out[meta_key][idx] = metadata_map[meta_key][sample_key]
        soft_probs[idx] = softmax_logits(vector[None, :], args.temperature)[0].astype(np.float16)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        soft_probs=soft_probs.astype(np.float16),
        labels=combined_labels,
        plant_ids=np.array(sorted_ids, dtype=object),
        latin_names=metadata_out['latin_names'],
        regions=metadata_out['regions'],
        urls=metadata_out['urls'],
        class_mapping_json=class_mapping_json,
        metadata=json.dumps({
            'teachers': [entry['path'] for entry in teacher_data],
            'weights': weights.tolist(),
            'temperature': args.temperature
        })
    )

    logger.info(f"Soft labels guardados en {output_path} (samples={soft_probs.shape[0]}, clases={soft_probs.shape[1]})")


if __name__ == '__main__':
    main()
