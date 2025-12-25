#!/usr/bin/env python3
"""Simple checkpoint quantization helpers (FP16)."""

import argparse
import logging
import shutil
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Quantize PyTorch checkpoint weights')
    parser.add_argument('--model', required=True, help='Checkpoint (.pt) a convertir')
    parser.add_argument('--output', required=True, help='Ruta del checkpoint resultante')
    parser.add_argument('--quantization_type', choices=['fp16', 'none'], default='fp16')
    return parser.parse_args()


def convert_state_dict_to_fp16(state_dict: dict):
    converted = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            converted[key] = value.half()
        else:
            converted[key] = value
    return converted


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.quantization_type == 'none':
        shutil.copyfile(model_path, output_path)
        logger.info(f'Checkpoint copiado sin cambios → {output_path}')
        return

    logger.info(f'Cargando checkpoint desde {model_path}')
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    logger.info('Convirtiendo tensores a FP16')
    checkpoint['model_state_dict'] = convert_state_dict_to_fp16(state_dict)
    checkpoint['quantization'] = 'fp16'

    torch.save(checkpoint, output_path)
    logger.info(f'Checkpoint FP16 guardado en {output_path}')


if __name__ == '__main__':
    main()
