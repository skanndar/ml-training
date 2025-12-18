#!/usr/bin/env python3
"""
split_dataset.py - Phase 0: Create Train/Val/Test Splits

Creates stratified train/val/test splits from the exported JSONL dataset.
Ensures:
- No data leakage between splits
- Stratified by class (latin_name) and region
- Balanced representation

Usage:
    python scripts/split_dataset.py --input ./data/dataset_raw.jsonl --output ./data
"""

import argparse
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_jsonl(filepath: str) -> list:
    """Load JSONL file into list of dicts."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading JSONL"):
            if line.strip():
                records.append(json.loads(line))
    return records


def create_stratified_splits(
    records: list,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    min_samples_per_class: int = 3
) -> tuple:
    """
    Create stratified train/val/test splits.

    Strategy:
    1. Deduplicate by URL (keep first occurrence)
    2. Group by class (latin_name)
    3. For classes with enough samples: stratified split
    4. For classes with few samples: all go to train
    """

    np.random.seed(seed)

    # Deduplicate by URL first (to prevent data leakage)
    logger.info(f"Deduplicating by URL...")
    seen_urls = set()
    unique_records = []
    duplicates = 0

    for record in records:
        url = record['image_url']
        if url not in seen_urls:
            seen_urls.add(url)
            unique_records.append(record)
        else:
            duplicates += 1

    logger.info(f"Removed {duplicates} duplicate URLs ({len(unique_records)} unique records)")

    # Group records by class
    class_records = defaultdict(list)
    for record in unique_records:
        class_name = record['latin_name']
        class_records[class_name].append(record)

    train_records = []
    val_records = []
    test_records = []

    stats = {
        'total_classes': len(class_records),
        'classes_stratified': 0,
        'classes_train_only': 0,
        'total_records': len(unique_records),
        'duplicates_removed': duplicates
    }

    logger.info(f"Creating splits for {len(class_records)} classes...")

    for class_name, class_recs in tqdm(class_records.items(), desc="Splitting classes"):
        n_samples = len(class_recs)

        if n_samples < min_samples_per_class:
            # Not enough samples for proper split - all go to train
            train_records.extend(class_recs)
            stats['classes_train_only'] += 1
        else:
            # Stratified split
            # First split: train vs (val + test)
            n_train = max(1, int(n_samples * train_ratio))
            n_val_test = n_samples - n_train

            if n_val_test < 2:
                # Can't split val/test further
                train_records.extend(class_recs[:n_train])
                val_records.extend(class_recs[n_train:])
            else:
                # Shuffle first
                shuffled = class_recs.copy()
                np.random.shuffle(shuffled)

                # Split
                train_records.extend(shuffled[:n_train])

                remaining = shuffled[n_train:]
                n_val = max(1, int(len(remaining) * (val_ratio / (val_ratio + test_ratio))))

                val_records.extend(remaining[:n_val])
                test_records.extend(remaining[n_val:])

            stats['classes_stratified'] += 1

    # Add split label to each record
    for record in train_records:
        record['split'] = 'train'
    for record in val_records:
        record['split'] = 'val'
    for record in test_records:
        record['split'] = 'test'

    # Shuffle final lists
    np.random.shuffle(train_records)
    np.random.shuffle(val_records)
    np.random.shuffle(test_records)

    logger.info(f"Train: {len(train_records):,} samples")
    logger.info(f"Val: {len(val_records):,} samples")
    logger.info(f"Test: {len(test_records):,} samples")

    return train_records, val_records, test_records, stats


def verify_no_leakage(train_records: list, val_records: list, test_records: list) -> bool:
    """Verify there's no data leakage between splits."""

    # Check for duplicate URLs
    train_urls = set(r['image_url'] for r in train_records)
    val_urls = set(r['image_url'] for r in val_records)
    test_urls = set(r['image_url'] for r in test_records)

    train_val_overlap = train_urls & val_urls
    train_test_overlap = train_urls & test_urls
    val_test_overlap = val_urls & test_urls

    if train_val_overlap:
        logger.warning(f"Train-Val overlap: {len(train_val_overlap)} URLs")
        return False

    if train_test_overlap:
        logger.warning(f"Train-Test overlap: {len(train_test_overlap)} URLs")
        return False

    if val_test_overlap:
        logger.warning(f"Val-Test overlap: {len(val_test_overlap)} URLs")
        return False

    logger.info("No data leakage detected")
    return True


def analyze_split_distribution(train_records: list, val_records: list, test_records: list) -> dict:
    """Analyze the distribution of splits."""

    analysis = {
        'train': {
            'total': len(train_records),
            'classes': len(set(r['latin_name'] for r in train_records)),
            'by_region': Counter(r['region'] for r in train_records),
            'by_source': Counter(r['image_source'] for r in train_records),
            'permissive_licenses': sum(1 for r in train_records if r.get('license_permissive', False))
        },
        'val': {
            'total': len(val_records),
            'classes': len(set(r['latin_name'] for r in val_records)),
            'by_region': Counter(r['region'] for r in val_records),
            'by_source': Counter(r['image_source'] for r in val_records),
            'permissive_licenses': sum(1 for r in val_records if r.get('license_permissive', False))
        },
        'test': {
            'total': len(test_records),
            'classes': len(set(r['latin_name'] for r in test_records)),
            'by_region': Counter(r['region'] for r in test_records),
            'by_source': Counter(r['image_source'] for r in test_records),
            'permissive_licenses': sum(1 for r in test_records if r.get('license_permissive', False))
        }
    }

    # Convert Counters to dicts
    for split in ['train', 'val', 'test']:
        analysis[split]['by_region'] = dict(analysis[split]['by_region'])
        analysis[split]['by_source'] = dict(analysis[split]['by_source'])

    return analysis


def save_splits(
    train_records: list,
    val_records: list,
    test_records: list,
    output_dir: str
):
    """Save splits to JSONL files."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save combined file with split labels
    all_records = train_records + val_records + test_records
    combined_file = output_path / 'dataset_splits.jsonl'

    with open(combined_file, 'w', encoding='utf-8') as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    logger.info(f"Combined splits saved to: {combined_file}")

    # Save individual split files
    for split_name, records in [('train', train_records), ('val', val_records), ('test', test_records)]:
        split_file = output_path / f'dataset_{split_name}.jsonl'
        with open(split_file, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        logger.info(f"  {split_name}: {split_file}")


def print_report(analysis: dict, stats: dict):
    """Print a summary report."""

    print("\n" + "=" * 70)
    print("                     DATASET SPLIT REPORT")
    print("=" * 70)

    print(f"\nTotal classes: {stats['total_classes']:,}")
    print(f"Classes with stratified split: {stats['classes_stratified']:,}")
    print(f"Classes train-only (few samples): {stats['classes_train_only']:,}")

    print("\n--- SPLIT SIZES ---")
    total = sum(analysis[s]['total'] for s in ['train', 'val', 'test'])
    for split in ['train', 'val', 'test']:
        count = analysis[split]['total']
        pct = count / total * 100 if total > 0 else 0
        classes = analysis[split]['classes']
        print(f"  {split:5s}: {count:>8,} samples ({pct:>5.1f}%) | {classes:,} classes")

    print("\n--- REGION DISTRIBUTION (Train) ---")
    for region, count in sorted(analysis['train']['by_region'].items(), key=lambda x: -x[1]):
        pct = count / analysis['train']['total'] * 100
        print(f"  {region:15s}: {count:>7,} ({pct:>5.1f}%)")

    print("\n--- SOURCE DISTRIBUTION (Train) ---")
    for source, count in sorted(analysis['train']['by_source'].items(), key=lambda x: -x[1]):
        pct = count / analysis['train']['total'] * 100
        print(f"  {source:15s}: {count:>7,} ({pct:>5.1f}%)")

    print("\n--- LICENSE DISTRIBUTION ---")
    for split in ['train', 'val', 'test']:
        permissive = analysis[split]['permissive_licenses']
        total_split = analysis[split]['total']
        pct = permissive / total_split * 100 if total_split > 0 else 0
        print(f"  {split:5s}: {permissive:>7,} permissive ({pct:>5.1f}%)")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Create train/val/test splits')
    parser.add_argument('--input', '-i', type=str,
                       default='./data/dataset_raw.jsonl',
                       help='Input JSONL file')
    parser.add_argument('--output', '-o', type=str,
                       default='./data',
                       help='Output directory')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Train split ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Test split ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--min-samples', type=int, default=3,
                       help='Minimum samples per class for stratified split')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        logger.error(f"Ratios must sum to 1.0, got {total_ratio}")
        return

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Load data
    logger.info(f"Loading data from: {args.input}")
    records = load_jsonl(args.input)
    logger.info(f"Loaded {len(records):,} records")

    # Create splits
    train_records, val_records, test_records, stats = create_stratified_splits(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        min_samples_per_class=args.min_samples
    )

    # Verify no leakage
    if not verify_no_leakage(train_records, val_records, test_records):
        logger.error("Data leakage detected! Please check the split logic.")
        return

    # Analyze distribution
    analysis = analyze_split_distribution(train_records, val_records, test_records)

    # Save splits
    save_splits(train_records, val_records, test_records, args.output)

    # Save analysis
    analysis_file = Path(args.output) / 'split_analysis.json'
    with open(analysis_file, 'w') as f:
        json.dump({
            'analysis': analysis,
            'stats': stats,
            'parameters': {
                'train_ratio': args.train_ratio,
                'val_ratio': args.val_ratio,
                'test_ratio': args.test_ratio,
                'seed': args.seed,
                'min_samples_per_class': args.min_samples
            },
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    logger.info(f"Analysis saved to: {analysis_file}")

    # Print report
    print_report(analysis, stats)

    logger.info("Split complete!")


if __name__ == '__main__':
    main()
