#!/usr/bin/env python3
"""
Create stratified train/val/test splits with optional region filtering and cache validation.

Key guarantees:
- All classes with ≥2 imágenes aparecen tanto en train como en val
- Clases con 1 imagen solo quedan en train (evita huecos en val)
- Funciona para dataset global o subsets regionales (EU_SW, EU_NORTH, ...)
"""

import argparse
import hashlib
import json
import logging
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - fallback si PyYAML no está disponible
    yaml = None


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create stratified dataset splits')
    parser.add_argument('--input', type=str, default='./data/dataset_raw.jsonl',
                        help='JSONL de entrada (default: data/dataset_raw.jsonl)')
    parser.add_argument('--output-prefix', type=str, default='./data/dataset',
                        help='Prefijo para los archivos de salida (sin _train/_val).')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Directorio del cache de imágenes. Si no se pasa, intenta usar config/paths.yaml o ./data/image_cache.')
    parser.add_argument('--skip-cache-filter', action='store_true',
                        help='No filtrar por imágenes cacheadas (usa todo el dataset).')
    parser.add_argument('--region', type=str, default=None,
                        help='Filtrar por región (ej: EU_SW o EU_NORTH,EU_EAST) antes de estratificar.')
    parser.add_argument('--val-ratio', type=float, default=0.10,
                        help='Porcentaje para validation (solo aplicado cuando hay ≥3 imágenes).')
    parser.add_argument('--test-ratio', type=float, default=0.10,
                        help='Porcentaje para test (solo aplicado cuando hay ≥3 imágenes).')
    parser.add_argument('--seed', type=int, default=42, help='Seed para reproducibilidad.')
    return parser.parse_args()


def resolve_cache_dir(arg_cache_dir: Optional[str]) -> Optional[Path]:
    """Determine cache directory with fallbacks."""
    if arg_cache_dir:
        return Path(arg_cache_dir)

    env_dir = os.environ.get('APLANTIDA_CACHE_DIR')
    if env_dir:
        return Path(env_dir)

    paths_yaml = Path('config/paths.yaml')
    if yaml and paths_yaml.exists():
        try:
            with open(paths_yaml, 'r', encoding='utf-8') as f:
                paths_cfg = yaml.safe_load(f) or {}
            data_cfg = paths_cfg.get('data', {}) or {}
            cache_dir = data_cfg.get('cache_dir')
            if cache_dir:
                return Path(cache_dir)
        except Exception as exc:  # pragma: no cover - logging informativo
            logger.warning(f'No se pudo leer config/paths.yaml: {exc}')

    return Path('./data/image_cache')


def url_to_filename(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest() + '.jpg'


def parse_region_filter(region_arg: Optional[str]) -> Optional[List[str]]:
    if not region_arg:
        return None
    regions = [r.strip() for r in region_arg.split(',') if r.strip()]
    return regions or None


def load_records(input_path: Path, cached: Optional[set], regions: Optional[List[str]]) -> List[dict]:
    records: List[dict] = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)

            if regions and rec.get('region') not in regions:
                continue

            if cached is not None:
                filename = url_to_filename(rec['image_url'])
                if filename not in cached:
                    continue

            records.append(rec)

    return records


def group_by_class(records: List[dict]) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    skipped = 0
    for rec in records:
        class_id = rec.get('class_idx') or rec.get('latin_name')
        if not class_id:
            skipped += 1
            continue
        groups[class_id].append(rec)

    if skipped:
        logger.warning(f"Registros sin class_idx/latin_name: {skipped}")
    return groups


def stratified_split(
    class_groups: Dict[str, List[dict]],
    val_ratio: float,
    test_ratio: float,
    seed: int
) -> Tuple[List[dict], List[dict], List[dict], Dict[str, int]]:
    rng = random.Random(seed)
    train_records: List[dict] = []
    val_records: List[dict] = []
    test_records: List[dict] = []

    stats = {
        'classes_1_image': 0,
        'classes_2_images': 0,
        'classes_3plus_images': 0
    }

    for class_id, images in class_groups.items():
        n = len(images)
        rng.shuffle(images)

        if n == 1:
            train_records.extend(images)
            stats['classes_1_image'] += 1
        elif n == 2:
            train_records.append(images[0])
            val_records.append(images[1])
            stats['classes_2_images'] += 1
        else:
            stats['classes_3plus_images'] += 1
            n_val = max(1, int(n * val_ratio))
            n_test = max(1, int(n * test_ratio))

            # Garantizar al menos 1 muestra en train
            n_train = max(1, n - n_val - n_test)

            train_records.extend(images[:n_train])
            val_records.extend(images[n_train:n_train + n_val])
            test_records.extend(images[n_train + n_val:])

    return train_records, val_records, test_records, stats


def save_jsonl(records: List[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def save_stats(
    stats_path: Path,
    info: Dict[str, object]
):
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f'Input file not found: {input_path}')

    regions = parse_region_filter(args.region)

    cache_dir = None
    cached = None
    if not args.skip_cache_filter:
        cache_dir = resolve_cache_dir(args.cache_dir)
        if cache_dir.exists():
            cached = {f.name for f in cache_dir.glob('*.jpg')}
            logger.info(f'Cache detectado: {cache_dir} ({len(cached):,} archivos)')
        else:
            logger.warning(f'Cache dir {cache_dir} no existe → usando dataset completo')

    logger.info(f'Cargando registros desde {input_path} (regiones={regions or "TODAS"})')
    records = load_records(input_path, cached, regions)
    logger.info(f'Registros válidos: {len(records):,}')

    if not records:
        raise RuntimeError('No hay registros después del filtrado. Verifica región o cache.')

    class_groups = group_by_class(records)
    logger.info(f'Clases únicas: {len(class_groups):,}')

    train_records, val_records, test_records, split_stats = stratified_split(
        class_groups, args.val_ratio, args.test_ratio, args.seed
    )

    stats = {
        'input_file': str(input_path),
        'region': regions,
        'total_records': len(records),
        'train_records': len(train_records),
        'val_records': len(val_records),
        'test_records': len(test_records),
        'train_classes': len(set(r.get('class_idx') or r.get('latin_name') for r in train_records)),
        'val_classes': len(set(r.get('class_idx') or r.get('latin_name') for r in val_records)),
        'test_classes': len(set(r.get('class_idx') or r.get('latin_name') for r in test_records)),
        'split_stats': split_stats,
        'cache_dir': str(cache_dir) if cache_dir else None,
        'cache_filtered': cached is not None
    }

    prefix_path = Path(args.output_prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)

    train_path = prefix_path.parent / f"{prefix_path.name}_train_stratified.jsonl"
    val_path = prefix_path.parent / f"{prefix_path.name}_val_stratified.jsonl"
    test_path = prefix_path.parent / f"{prefix_path.name}_test_stratified.jsonl"

    save_jsonl(train_records, train_path)
    save_jsonl(val_records, val_path)
    save_jsonl(test_records, test_path)

    stats_path = prefix_path.parent / f"{prefix_path.name}_stratified_stats.json"
    save_stats(stats_path, stats)

    logger.info('===============================================================')
    logger.info('Resumen estratificado:')
    logger.info(f"  Train: {len(train_records):,} imágenes")
    logger.info(f"  Val:   {len(val_records):,} imágenes")
    logger.info(f"  Test:  {len(test_records):,} imágenes")
    logger.info(f"  Clases Train/Val/Test: {stats['train_classes']:,} / {stats['val_classes']:,} / {stats['test_classes']:,}")
    logger.info(f"  Guardado en: {train_path}, {val_path}, {test_path}")
    logger.info(f"  Stats: {stats_path}")
    logger.info('===============================================================')


if __name__ == '__main__':
    main()
