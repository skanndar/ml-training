#!/usr/bin/env python3
"""
create_regional_subset.py - Create Regional Dataset Subset

Filters the train dataset by region for regional teacher training.
Used for teacher_regional model (EU_SW focus).

Usage:
    python scripts/create_regional_subset.py --region EU_SW --input ./data/dataset_train.jsonl --output ./data/dataset_eu_sw_train.jsonl
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_jsonl(filepath: str) -> list:
    """Load JSONL file."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading JSONL"):
            if line.strip():
                records.append(json.loads(line))
    return records


def filter_by_region(records: list, region: str) -> list:
    """Filter records by region."""
    filtered = [r for r in records if r.get('region') == region]
    return filtered


def analyze_subset(records: list) -> dict:
    """Analyze the subset."""
    analysis = {
        'total_records': len(records),
        'num_classes': len(set(r['latin_name'] for r in records)),
        'by_region': Counter(r['region'] for r in records),
        'by_source': Counter(r['image_source'] for r in records),
        'permissive_licenses': sum(1 for r in records if r.get('license_permissive', False))
    }

    # Class distribution
    class_counts = Counter(r['latin_name'] for r in records)
    analysis['avg_samples_per_class'] = sum(class_counts.values()) / len(class_counts)
    analysis['min_samples_per_class'] = min(class_counts.values())
    analysis['max_samples_per_class'] = max(class_counts.values())

    # Convert Counters to dicts
    analysis['by_region'] = dict(analysis['by_region'])
    analysis['by_source'] = dict(analysis['by_source'])

    return analysis


def save_jsonl(records: list, output_path: str):
    """Save records to JSONL."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    logger.info(f"Saved {len(records)} records to: {output_path}")


def print_report(analysis: dict, region: str):
    """Print analysis report."""
    print("\n" + "="*70)
    print(f"              REGIONAL SUBSET REPORT: {region}")
    print("="*70)

    print(f"\nTotal records:         {analysis['total_records']:,}")
    print(f"Number of classes:     {analysis['num_classes']:,}")
    print(f"Avg samples/class:     {analysis['avg_samples_per_class']:.1f}")
    print(f"Min samples/class:     {analysis['min_samples_per_class']}")
    print(f"Max samples/class:     {analysis['max_samples_per_class']}")

    print("\n--- REGION DISTRIBUTION ---")
    for region_name, count in sorted(analysis['by_region'].items(), key=lambda x: -x[1]):
        pct = count / analysis['total_records'] * 100
        print(f"  {region_name:15s}: {count:>7,} ({pct:>5.1f}%)")

    print("\n--- SOURCE DISTRIBUTION ---")
    for source, count in sorted(analysis['by_source'].items(), key=lambda x: -x[1]):
        pct = count / analysis['total_records'] * 100
        print(f"  {source:15s}: {count:>7,} ({pct:>5.1f}%)")

    permissive = analysis['permissive_licenses']
    total = analysis['total_records']
    pct_permissive = permissive / total * 100 if total > 0 else 0
    print(f"\n--- LICENSE DISTRIBUTION ---")
    print(f"  Permissive licenses: {permissive:>7,} ({pct_permissive:>5.1f}%)")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Create regional dataset subset')
    parser.add_argument('--region', '-r', type=str, required=True,
                       help='Region filter (EU_SW, EU_NORTH, etc.)')
    parser.add_argument('--input', '-i', type=str,
                       default='./data/dataset_train.jsonl',
                       help='Input JSONL file (train set)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output JSONL file (auto-generated if not specified)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Auto-generate output filename if not specified
    if not args.output:
        region_lower = args.region.lower()
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"dataset_{region_lower}_train.jsonl")

    # Load data
    logger.info(f"Loading data from: {args.input}")
    records = load_jsonl(args.input)
    logger.info(f"Loaded {len(records):,} records")

    # Filter by region
    logger.info(f"Filtering by region: {args.region}")
    filtered_records = filter_by_region(records, args.region)
    logger.info(f"Found {len(filtered_records):,} records in {args.region}")

    if len(filtered_records) == 0:
        logger.error(f"No records found for region {args.region}")
        logger.info("Available regions:")
        regions = Counter(r['region'] for r in records)
        for region, count in sorted(regions.items(), key=lambda x: -x[1]):
            logger.info(f"  {region}: {count:,}")
        return

    # Analyze
    analysis = analyze_subset(filtered_records)

    # Save
    save_jsonl(filtered_records, args.output)

    # Save analysis
    analysis_file = Path(args.output).parent / f"analysis_{args.region.lower()}.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"Analysis saved to: {analysis_file}")

    # Print report
    print_report(analysis, args.region)

    logger.info("Regional subset creation complete!")


if __name__ == '__main__':
    main()
