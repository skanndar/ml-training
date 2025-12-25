#!/usr/bin/env python3
"""Export PyTorch checkpoint → ONNX → TF.js graph model."""

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path

import onnx
import onnx_tf.backend as onnx_backend
import timm
import torch
import yaml

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_utils import TemperatureScaledModel, load_class_mapping  # type: ignore

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Export PyTorch checkpoint to TF.js')
    parser.add_argument('--config', required=True, help='Config YAML (student)')
    parser.add_argument('--model', required=True, help='Checkpoint (.pt) a exportar')
    parser.add_argument('--output-dir', required=True, help='Directorio final TF.js')
    parser.add_argument('--class-mapping', help='JSON con mapping clase→índice')
    parser.add_argument('--train-jsonl', help='JSONL para derivar mapping si no existe archivo')
    parser.add_argument('--onnx-output', help='Ruta opcional para guardar ONNX intermedio')
    parser.add_argument('--saved-model-dir', help='Ruta opcional SavedModel (default: dentro de output)')
    parser.add_argument('--image-size', type=int, help='Sobrescribe image_size del config (opcional)')
    parser.add_argument('--quantization', choices=['none', 'float16', 'uint8'], default='float16')
    parser.add_argument('--force', action='store_true', help='Sobrescribe output dir si existe')
    return parser.parse_args()


def build_model(config_path: Path, checkpoint_path: Path, num_classes: int):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    model = timm.create_model(
        cfg['model']['name'],
        pretrained=False,
        num_classes=num_classes
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    missing = model.load_state_dict(state_dict, strict=False)
    if missing and (missing.missing_keys or missing.unexpected_keys):
        logger.warning('Missing keys: %s / Unexpected keys: %s', missing.missing_keys, missing.unexpected_keys)

    temperature = checkpoint.get('temperature')
    if temperature is not None:
        model = TemperatureScaledModel(model, float(temperature))

    fp16_source = checkpoint.get('quantization') == 'fp16'
    if fp16_source:
        logger.info('Checkpoint marcado como FP16 → se exportará en float32 (con cuantización posterior)')

    model = model.float()
    model.eval()
    return model, cfg, temperature, fp16_source


def export_to_onnx(model: torch.nn.Module, onnx_path: Path, image_size: int):
    dummy = torch.randn(1, 3, image_size, image_size)

    logger.info(f'Exportando ONNX → {onnx_path}')
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=13,
        input_names=['input'],
        output_names=['logits'],
        dynamic_axes={'input': {0: 'batch_size'}, 'logits': {0: 'batch_size'}}
    )
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    return onnx_model


def convert_to_saved_model(onnx_model, saved_model_dir: Path):
    logger.info(f'Convirtiendo ONNX → SavedModel ({saved_model_dir})')
    saved_model_dir.mkdir(parents=True, exist_ok=True)
    tf_rep = onnx_backend.prepare(onnx_model)
    tf_rep.export_graph(str(saved_model_dir))


def run_tfjs_converter(saved_model_dir: Path, output_dir: Path, quantization: str):
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    converter = Path(sys.executable).with_name('tensorflowjs_converter')
    cmd = [
        str(converter),
        '--input_format', 'tf_saved_model',
        '--output_format', 'tfjs_graph_model'
    ]

    if quantization == 'float16':
        cmd.extend(['--quantize_float16', '*'])
    elif quantization == 'uint8':
        cmd.extend(['--quantize_uint8', '*'])

    cmd.extend([str(saved_model_dir), str(output_dir)])
    logger.info('Ejecutando tensorflowjs_converter: %s', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def write_metadata(output_dir: Path, metadata: dict):
    meta_path = output_dir / 'export_metadata.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f'Metadata guardada en {meta_path}')


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.force:
        raise FileExistsError(f'{output_dir} ya existe. Usa --force para sobrescribir.')

    class_mapping = load_class_mapping(args.class_mapping, args.train_jsonl)
    num_classes = len(class_mapping)

    checkpoint_path = Path(args.model)
    config_path = Path(args.config)
    model, cfg, temperature, fp16_source = build_model(config_path, checkpoint_path, num_classes)

    image_size = args.image_size or cfg['data'].get('image_size', 224)

    intermediate_dir = output_dir / '_intermediate'
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = Path(args.onnx_output) if args.onnx_output else (intermediate_dir / 'model.onnx')
    saved_model_dir = Path(args.saved_model_dir) if args.saved_model_dir else (intermediate_dir / 'saved_model')

    onnx_model = export_to_onnx(model, onnx_path, image_size)
    convert_to_saved_model(onnx_model, saved_model_dir)
    run_tfjs_converter(saved_model_dir, output_dir, args.quantization)

    shard_sizes = []
    for file in output_dir.glob('*.bin'):
        shard_sizes.append({'file': file.name, 'size_bytes': file.stat().st_size})

    metadata = {
        'checkpoint': str(checkpoint_path),
        'config': str(config_path),
        'num_classes': num_classes,
        'image_size': image_size,
        'temperature': float(temperature) if temperature is not None else None,
        'onnx_path': str(onnx_path),
        'saved_model_dir': str(saved_model_dir),
        'tfjs_dir': str(output_dir),
        'quantization': args.quantization,
        'shards': shard_sizes,
        'source_fp16': fp16_source
    }
    write_metadata(output_dir, metadata)


if __name__ == '__main__':
    main()
