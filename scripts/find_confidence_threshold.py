#!/usr/bin/env python3
"""Analiza la calibración y propone un umbral de confianza."""

import argparse
import json
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Find confidence threshold for "no concluyente"')
    parser.add_argument('--predictions', required=True, help='Archivo JSON generado por calibrate_model.py')
    parser.add_argument('--output', required=True, help='Archivo JSON para guardar el análisis')
    parser.add_argument('--target-accuracy', type=float, default=0.95,
                        help='Accuracy mínimo deseado para aceptar predicciones (default: 0.95)')
    parser.add_argument('--min-threshold', type=float, default=0.4)
    parser.add_argument('--max-threshold', type=float, default=0.95)
    parser.add_argument('--step', type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    data = json.loads(Path(args.predictions).read_text())
    confidences = np.array(data['confidences'])
    correct = np.array(data['correct']).astype(bool)

    thresholds = np.arange(args.min_threshold, args.max_threshold + 1e-6, args.step)
    rows = []
    best = None

    for thr in thresholds:
        mask = confidences >= thr
        coverage = float(mask.sum()) / len(confidences)
        if mask.sum() == 0:
            acc = None
        else:
            acc = float(correct[mask].mean())
        rows.append({
            'threshold': float(thr),
            'coverage': coverage,
            'accuracy': acc
        })
        if acc is not None and acc >= args.target_accuracy:
            if best is None or coverage > best['coverage']:
                best = {'threshold': float(thr), 'coverage': coverage, 'accuracy': acc}

    if best is None:
        best = max((row for row in rows if row['accuracy'] is not None), key=lambda r: r['accuracy'])
        logger.warning('No threshold reach target accuracy %.2f; picking best accuracy %.3f', args.target_accuracy, best['accuracy'])
    else:
        logger.info('Threshold %.2f achieves accuracy %.3f with coverage %.3f',
                    best['threshold'], best['accuracy'], best['coverage'])

    payload = {
        'predictions_file': args.predictions,
        'target_accuracy': args.target_accuracy,
        'analysis': rows,
        'recommended': best
    }
    Path(args.output).write_text(json.dumps(payload, indent=2))
    logger.info(f"Análisis guardado en {args.output}")


if __name__ == '__main__':
    main()
