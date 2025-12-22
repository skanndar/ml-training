#!/usr/bin/env python3
"""
Map existing cached images to medium URLs without re-downloading.

This script:
1. Scans the existing cache to find which images are already downloaded
2. Updates the JSONL files to use medium URLs for images that exist in cache
3. Leaves missing images as medium URLs (will be downloaded as needed)

This way you keep the images already downloaded (avoiding rate limits)
while new downloads will be medium-sized (20x faster).
"""

import hashlib
import json
import sys
from pathlib import Path
from collections import defaultdict

def url_to_filename(url: str) -> str:
    """Convert URL to cache filename (same as streaming_dataset.py)"""
    url_hash = hashlib.md5(url.encode()).hexdigest()
    return f"{url_hash}.jpg"

def main():
    # Paths
    cache_dir = Path("/media/skanndar/2TB1/aplantida-ml/image_cache")
    train_jsonl = Path("data/dataset_train.jsonl")
    val_jsonl = Path("data/dataset_val.jsonl")

    print("=" * 70)
    print("  Mapping Existing Cache to Medium URLs")
    print("=" * 70)
    print()

    # Step 1: Build a set of cached image hashes
    print("Step 1: Scanning existing cache...")
    cached_files = set()
    if cache_dir.exists():
        for img_file in cache_dir.glob("*.jpg"):
            cached_files.add(img_file.name)

    print(f"  Found {len(cached_files):,} cached images")
    print()

    # Step 2: Build reverse mapping from hash to original URL
    print("Step 2: Building URL mappings...")

    # We need to know which original URL corresponds to each cached file
    # by checking both original and medium URLs
    original_to_medium = {}
    medium_to_cached = set()

    # Process each JSONL file
    for jsonl_path in [train_jsonl, val_jsonl]:
        print(f"  Processing {jsonl_path}...")

        if not jsonl_path.exists():
            print(f"    ⚠️  {jsonl_path} not found, skipping")
            continue

        with open(jsonl_path, 'r') as f:
            for line in f:
                record = json.loads(line.strip())
                original_url = record['image_url']

                # Generate medium URL
                medium_url = original_url.replace('/original.jpg', '/medium.jpg')

                # Check if we have this image in cache (as original)
                original_hash = url_to_filename(original_url)

                if original_hash in cached_files:
                    # We have this image cached as original
                    # We'll KEEP using original URL to avoid re-download
                    pass  # Keep original URL
                else:
                    # We don't have it, will use medium URL
                    original_to_medium[original_url] = medium_url

    print(f"  Images already cached (will keep original): {len(cached_files):,}")
    print(f"  Images not cached (will use medium): {len(original_to_medium):,}")
    print()

    # Step 3: Update JSONL files
    print("Step 3: Updating JSONL files...")

    for jsonl_path in [train_jsonl, val_jsonl]:
        if not jsonl_path.exists():
            continue

        # Create backup
        backup_path = jsonl_path.with_suffix('.jsonl.backup_before_medium')
        if not backup_path.exists():
            print(f"  Creating backup: {backup_path}")
            import shutil
            shutil.copy2(jsonl_path, backup_path)

        # Read all records
        records = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                records.append(json.loads(line.strip()))

        # Update URLs
        kept_original = 0
        changed_to_medium = 0

        for record in records:
            original_url = record['image_url']
            original_hash = url_to_filename(original_url)

            if original_hash in cached_files:
                # Keep original URL (already cached)
                kept_original += 1
            else:
                # Change to medium URL (not cached, will download medium)
                medium_url = original_url.replace('/original.jpg', '/medium.jpg')
                record['image_url'] = medium_url
                changed_to_medium += 1

        # Write updated file
        with open(jsonl_path, 'w') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')

        print(f"  {jsonl_path.name}:")
        print(f"    - Kept original URLs (cached): {kept_original:,}")
        print(f"    - Changed to medium URLs (not cached): {changed_to_medium:,}")
        print()

    print("=" * 70)
    print("  ✅ Done!")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  • {len(cached_files):,} images will use existing cache (original size)")
    print(f"  • ~{len(original_to_medium):,} missing images will download as medium (20x faster)")
    print()
    print("Benefits:")
    print("  • No rate limit issues (reusing existing downloads)")
    print("  • New downloads are 20x faster (medium vs original)")
    print("  • No wasted disk space")
    print()
    print("Next steps:")
    print("  1. Stop training when epoch completes:")
    print("     pkill -SIGTERM -f train_teacher.py")
    print()
    print("  2. Wait for checkpoint to save:")
    print("     sleep 10")
    print()
    print("  3. Resume training:")
    print("     cd /home/skanndar/SynologyDrive/local/aplantida/ml-training")
    print("     source venv/bin/activate")
    print("     nohup python3 scripts/train_teacher.py \\")
    print("         --config config/teacher_global.yaml \\")
    print("         --resume checkpoints/teacher_global/last_checkpoint.pt \\")
    print("         > training.log 2>&1 &")
    print()
    print("To restore original URLs:")
    print(f"  cp {backup_path} {train_jsonl}")
    print(f"  cp {val_jsonl}.backup_before_medium {val_jsonl}")
    print()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
