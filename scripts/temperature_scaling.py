#!/usr/bin/env python3
"""Temperature scaling for calibration (ECE reduction)."""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.calibrate_model import compute_calibration_stats  # type: ignore
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
    parser = argparse.ArgumentParser(description='Temperature scaling for calibration')
    parser.add_argument('--config', required=True, help='Config YAML (student/teacher)')
    parser.add_argument('--model', required=True, help='Checkpoint a calibrar (.pt)')
    parser.add_argument('--val_jsonl', required=True, help='JSONL con split de validación')
    parser.add_argument('--output', '--output-temperature', dest='output', required=True,
                        help='Ruta para guardar checkpoint con temperatura aplicada')
    parser.add_argument('--class-mapping', help='JSON con mapping clase→índice')
    parser.add_argument('--train-jsonl', help='JSONL para derivar mapping si no existe archivo')
    parser.add_argument('--save-class-mapping', help='Ruta opcional para guardar mapping derivado')
    parser.add_argument('--batch-size', type=int, help='Batch size de inferencia')
    parser.add_argument('--device', help='cuda/cpu (auto si no se especifica)')
    parser.add_argument('--min-temperature', type=float, default=0.5,
                        help='Temperatura mínima a evaluar (default: 0.5)')
    parser.add_argument('--max-temperature', type=float, default=3.5,
                        help='Temperatura máxima a evaluar (default: 3.5)')
    parser.add_argument('--num-temperatures', type=int, default=20,
                        help='Cantidad de temperaturas en la búsqueda gruesa (default: 20)')
    parser.add_argument('--refine-steps', type=int, default=10,
                        help='Pasos adicionales alrededor de la mejor temperatura (default: 10)')
    parser.add_argument('--refine-range', type=float, default=0.2,
                        help='Porción del rango total usada para el refinamiento (default: 0.2)')
    parser.add_argument('--bins', type=int, default=15, help='Bins para métricas de calibración (default: 15)')
    parser.add_argument('--metrics-output', help='Archivo JSON para guardar métricas antes/después')
    return parser.parse_args()


def gather_logits(model, dataloader, device):
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc='Recolectando logits'):
            images = images.to(device)
            outputs = model(images)
            logits_list.append(outputs.detach().cpu())
            labels_list.append(labels.detach().cpu())

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return logits, labels


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, bins: int):
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        loss = criterion(logits, labels).item()
        probs = torch.softmax(logits, dim=1)
        conf, preds = probs.max(dim=1)
        correct = preds.eq(labels)
        ece, mce, _ = compute_calibration_stats(
            conf.cpu().numpy(),
            correct.cpu().numpy(),
            bins
        )
    return loss, ece, mce


def _evaluate_temperature(logits: torch.Tensor, labels: torch.Tensor, temp: float, bins: int):
    scaled_logits = logits / temp
    nll, ece, mce = compute_metrics(scaled_logits, labels, bins)
    return {
        'temperature': float(temp),
        'nll': float(nll),
        'ece': float(ece),
        'mce': float(mce)
    }


def search_temperature(logits: torch.Tensor, labels: torch.Tensor, bins: int,
                       min_temp: float, max_temp: float,
                       num_temps: int, refine_steps: int, refine_range: float):
    device = logits.device
    temps = torch.linspace(min_temp, max_temp, steps=max(2, num_temps), device=device)
    best = None
    history = []

    for temp in temps:
        result = _evaluate_temperature(logits, labels, float(temp.item()), bins)
        history.append(result)
        if best is None or result['ece'] < best['ece']:
            best = result

    if refine_steps > 0 and refine_range > 0:
        span = (max_temp - min_temp) * refine_range
        if span > 0:
            ref_min = max(min_temp, best['temperature'] - span / 2)
            ref_max = min(max_temp, best['temperature'] + span / 2)
            refine_temps = torch.linspace(ref_min, ref_max, steps=max(2, refine_steps), device=device)
            for temp in refine_temps:
                result = _evaluate_temperature(logits, labels, float(temp.item()), bins)
                history.append(result)
                if result['ece'] < best['ece']:
                    best = result

    return best, history


def main():
    args = parse_args()

    class_mapping = load_class_mapping(args.class_mapping, args.train_jsonl)
    save_class_mapping(class_mapping, args.save_class_mapping)

    dataloader, _, config = build_eval_dataloader(
        config_path=args.config,
        jsonl_path=args.val_jsonl,
        class_mapping=class_mapping,
        batch_size=args.batch_size,
        mode='val'
    )

    device = resolve_device(args.device)
    model = load_model_from_checkpoint(config, args.model, len(class_mapping), device)

    logits_cpu, labels_cpu = gather_logits(model, dataloader, device)
    logits = logits_cpu.to(device)
    labels = labels_cpu.to(device)

    nll_before, ece_before, mce_before = compute_metrics(logits, labels, args.bins)
    logger.info(f"Antes: NLL={nll_before:.4f}, ECE={ece_before:.4f}, MCE={mce_before:.4f}")

    best_result, search_history = search_temperature(
        logits,
        labels,
        args.bins,
        args.min_temperature,
        args.max_temperature,
        args.num_temperatures,
        args.refine_steps,
        args.refine_range
    )

    best_temperature = best_result['temperature']
    logger.info(f"Temperatura óptima: {best_temperature:.4f} (ECE={best_result['ece']:.4f})")

    scaled_logits = logits / best_temperature
    nll_after, ece_after, mce_after = compute_metrics(scaled_logits, labels, args.bins)
    logger.info(f"Después: NLL={nll_after:.4f}, ECE={ece_after:.4f}, MCE={mce_after:.4f}")

    checkpoint = torch.load(args.model, map_location='cpu')
    checkpoint['temperature'] = float(best_temperature)
    checkpoint['temperature_metrics'] = {
        'nll_before': float(nll_before),
        'nll_after': float(nll_after),
        'ece_before': float(ece_before),
        'ece_after': float(ece_after),
        'mce_before': float(mce_before),
        'mce_after': float(mce_after),
        'bins': int(args.bins),
        'samples': int(logits.shape[0])
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    logger.info(f"Checkpoint con temperatura guardado en {output_path}")

    if args.metrics_output:
        metrics_payload = {
            'model': args.model,
            'output_checkpoint': str(output_path),
            'temperature': float(best_temperature),
            'nll_before': float(nll_before),
            'nll_after': float(nll_after),
            'ece_before': float(ece_before),
            'ece_after': float(ece_after),
            'mce_before': float(mce_before),
            'mce_after': float(mce_after),
            'bins': int(args.bins),
            'samples': int(logits.shape[0]),
            'search_history': search_history
        }
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics_payload, indent=2))
        logger.info(f"Métricas guardadas en {metrics_path}")


if __name__ == '__main__':
    main()
