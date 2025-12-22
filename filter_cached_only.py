#!/usr/bin/env python3
"""
Filter JSONL to include ONLY images that are already cached.

This creates new dataset files with only successfully cached images,
avoiding all the download failures and blank images.
"""

import hashlib
import json
import sys
from pathlib import Path

def url_to_filename(url: str) -> str:
    """Convert URL to cache filename (same as streaming_dataset.py)"""
    url_hash = hashlib.md5(url.encode()).hexdigest()
    return f"{url_hash}.jpg"

def main():
    cache_dir = Path("/media/skanndar/2TB1/aplantida-ml/image_cache")

    print("=" * 70)
    print("  Filter Dataset to Cached Images Only")
    print("=" * 70)
    print()

    # Build set of cached images
    print("Scanning cache...")
    cached_files = set()
    if cache_dir.exists():
        for img_file in cache_dir.glob("*.jpg"):
            cached_files.add(img_file.name)

    print(f"Found {len(cached_files):,} cached images")
    print()

    # Process each dataset
    for dataset_name in ['train', 'val']:
        jsonl_path = Path(f"data/dataset_{dataset_name}.jsonl")
        output_path = Path(f"data/dataset_{dataset_name}_cached_only.jsonl")

        if not jsonl_path.exists():
            print(f"⚠️  {jsonl_path} not found, skipping")
            continue

        print(f"Processing {dataset_name} dataset...")

        # Read all records
        records = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                records.append(json.loads(line.strip()))

        # Filter to cached only
        cached_records = []
        for record in records:
            url = record['image_url']
            cache_filename = url_to_filename(url)

            if cache_filename in cached_files:
                cached_records.append(record)

        # Write filtered dataset
        with open(output_path, 'w') as f:
            for record in cached_records:
                f.write(json.dumps(record) + '\n')

        print(f"  Original: {len(records):,} images")
        print(f"  Cached: {len(cached_records):,} images ({len(cached_records)/len(records)*100:.1f}%)")
        print(f"  Saved to: {output_path}")
        print()

    print("=" * 70)
    print("  ✅ Done!")
    print("=" * 70)
    print()
    print("Summary:")
    print("  • New datasets contain ONLY successfully cached images")
    print("  • NO blank images")
    print("  • NO download failures")
    print("  • Training will be faster and more accurate")
    print()
    print("To use the filtered datasets:")
    print("  1. Edit config/teacher_global.yaml")
    print("  2. Change:")
    print("     train_jsonl: './data/dataset_train_cached_only.jsonl'")
    print("     val_jsonl: './data/dataset_val_cached_only.jsonl'")
    print("  3. Restart training with --resume")
    print()

if __name__ == '__main__':
    main()
