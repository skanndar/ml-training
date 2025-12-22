#!/usr/bin/env python3
"""
Create a small test dataset for rapid prototyping.

This creates a balanced test set with:
- 70% cached images (fast loading)
- 30% uncached images (to test downloads and rate limiting)
- Stratified sampling across classes
"""

import json
import hashlib
import random
from pathlib import Path
from collections import defaultdict


def url_to_filename(url: str) -> str:
    """Convert URL to cache filename."""
    return hashlib.md5(url.encode()).hexdigest() + ".jpg"


def main():
    """Create test dataset with mixed cached/uncached images."""
    print("=" * 70)
    print("  Creating Test Dataset (with Download Testing)")
    print("=" * 70)
    print()

    # Parameters
    images_per_class = 10  # 10 images per class for better testing
    min_class_size = 5     # Only include classes with at least 5 images
    max_classes = 100      # 100 classes = ~1000 images total
    target_cache_rate = 0.70  # 70% cached, 30% uncached

    # Paths
    cache_dir = Path("/media/skanndar/2TB1/aplantida-ml/image_cache")
    # Use RAW dataset which has uncached images from new sources
    input_file = Path("data/dataset_raw.jsonl")
    output_train = Path("data/dataset_test_train.jsonl")
    output_val = Path("data/dataset_test_val.jsonl")

    # Load full dataset
    print(f"Loading dataset from {input_file}")
    with open(input_file, encoding='utf-8') as f:
        records = [json.loads(line) for line in f]

    print(f"Total records: {len(records):,}")
    print()

    # Group by class and check cache status
    class_records = defaultdict(lambda: {'cached': [], 'uncached': []})
    total_cached = 0

    for rec in records:
        url = rec['image_url']
        latin_name = rec['latin_name']
        filename = url_to_filename(url)

        # Check if cached
        is_cached = (cache_dir / filename).exists()
        if is_cached:
            total_cached += 1
            class_records[latin_name]['cached'].append(rec)
        else:
            class_records[latin_name]['uncached'].append(rec)

    print(f"Cache status:")
    print(f"  Cached: {total_cached:,} ({total_cached/len(records)*100:.1f}%)")
    print(f"  Uncached: {len(records) - total_cached:,}")
    print()

    # Filter classes with enough images
    valid_classes = {}
    for name, recs in class_records.items():
        total = len(recs['cached']) + len(recs['uncached'])
        if total >= min_class_size:
            valid_classes[name] = recs

    print(f"Classes with >= {min_class_size} images: {len(valid_classes):,}")

    # Sort by cache diversity (prefer classes with both cached and uncached)
    def cache_diversity_score(class_data):
        cached_count = len(class_data['cached'])
        uncached_count = len(class_data['uncached'])
        total = cached_count + uncached_count
        # Prefer classes with both cached and uncached images
        if cached_count > 0 and uncached_count > 0:
            return total + 100  # Bonus for diversity
        return total

    classes_by_diversity = sorted(
        valid_classes.items(),
        key=lambda x: cache_diversity_score(x[1]),
        reverse=True
    )

    # Take top N classes
    selected_classes = dict(classes_by_diversity[:max_classes])
    print(f"Selected {len(selected_classes)} classes with good cache diversity")
    print()

    # Sample from each class (mix of cached and uncached)
    all_samples = []

    for recs in selected_classes.values():
        cached = recs['cached']
        uncached = recs['uncached']

        # Calculate how many of each type to sample
        n_to_sample = min(images_per_class, len(cached) + len(uncached))
        n_cached = int(n_to_sample * target_cache_rate)
        n_uncached = n_to_sample - n_cached

        # Sample
        samples = []
        if len(cached) >= n_cached:
            samples.extend(random.sample(cached, n_cached))
        else:
            samples.extend(cached)
            n_uncached += (n_cached - len(cached))

        if len(uncached) >= n_uncached:
            samples.extend(random.sample(uncached, n_uncached))
        else:
            samples.extend(uncached)

        all_samples.extend(samples)

    # Shuffle
    random.shuffle(all_samples)

    # 80/20 train/val split
    n_train = int(len(all_samples) * 0.8)
    train_records = all_samples[:n_train]
    val_records = all_samples[n_train:]

    # Calculate cache coverage
    train_cached = sum(
        1 for r in train_records
        if (cache_dir / url_to_filename(r['image_url'])).exists()
    )
    val_cached = sum(
        1 for r in val_records
        if (cache_dir / url_to_filename(r['image_url'])).exists()
    )

    # Write outputs
    print("Writing test datasets:")
    print()

    with open(output_train, 'w', encoding='utf-8') as f:
        for rec in train_records:
            f.write(json.dumps(rec) + '\n')

    with open(output_val, 'w', encoding='utf-8') as f:
        for rec in val_records:
            f.write(json.dumps(rec) + '\n')

    train_uncached = len(train_records) - train_cached
    val_uncached = len(val_records) - val_cached

    print(f"✅ Train: {output_train}")
    print(f"   Total: {len(train_records):,}")
    print(f"   Cached: {train_cached:,} ({train_cached/len(train_records)*100:.1f}%)")
    print(f"   Uncached: {train_uncached:,} ({train_uncached/len(train_records)*100:.1f}%)")
    print()

    print(f"✅ Val: {output_val}")
    print(f"   Total: {len(val_records):,}")
    print(f"   Cached: {val_cached:,} ({val_cached/len(val_records)*100:.1f}%)")
    print(f"   Uncached: {val_uncached:,} ({val_uncached/len(val_records)*100:.1f}%)")
    print()

    print(f"Classes: {len(selected_classes)}")
    print()

    # Expected downloads
    total_uncached = train_uncached + val_uncached
    print(f"⚠️  Download Testing:")
    print(f"   Will attempt to download {total_uncached} uncached images")
    print(f"   This will test rate limiting and fallback behavior")
    print()

    # Calculate expected training time
    batches_per_epoch = len(train_records) // 4
    seconds_per_epoch = batches_per_epoch / 2.5

    print(f"Expected training time (15 epochs):")
    print(f"  Batches/epoch: {batches_per_epoch}")
    print(f"  Time/epoch: ~{seconds_per_epoch/60:.0f} min")
    print(f"  Total (15 epochs): ~{seconds_per_epoch*15/3600:.1f} hours")
    print()

    print("=" * 70)
    print("Next steps:")
    print()
    print("1. Config is already updated to use test dataset")
    print()
    print("2. Delete old checkpoints:")
    print("   rm -rf checkpoints/teacher_global")
    print()
    print("3. Start training:")
    print("   python3 scripts/train_teacher.py --config config/teacher_global.yaml")
    print()
    print("4. Monitor downloads in real-time:")
    print("   tail -f training_test.log | grep -E 'download|rate|429|cache'")
    print("=" * 70)


if __name__ == "__main__":
    random.seed(42)  # Reproducible
    main()
