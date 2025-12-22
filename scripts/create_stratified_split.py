#!/usr/bin/env python3
"""
Create stratified train/val/test split ensuring:
- All classes appear in both train and validation (if they have enough images)
- Classes with <3 images go only to train
- Proper distribution maintained per class
"""

import json
import random
from pathlib import Path
from collections import defaultdict

def main():
    # Load all cached records
    input_file = Path("data/dataset_raw.jsonl")

    # Build cache set
    import hashlib
    cache_dir = Path("/media/skanndar/2TB1/aplantida-ml/image_cache")
    cached = {f.name for f in cache_dir.glob("*.jpg")}

    def url_to_filename(url):
        return hashlib.md5(url.encode()).hexdigest() + ".jpg"

    # Load and filter to cached only
    records = []
    with open(input_file) as f:
        for line in f:
            rec = json.loads(line)
            if url_to_filename(rec['image_url']) in cached:
                records.append(rec)

    print(f"Loaded {len(records):,} cached records")

    # Group by class
    classes = defaultdict(list)
    for rec in records:
        classes[rec['class_idx']].append(rec)

    print(f"Found {len(classes):,} unique classes")

    # Analyze distribution
    class_sizes = [len(imgs) for imgs in classes.values()]
    print(f"\nClass distribution:")
    print(f"  Min: {min(class_sizes)}")
    print(f"  Max: {max(class_sizes)}")
    print(f"  Avg: {sum(class_sizes)/len(class_sizes):.1f}")
    print(f"  Classes with 1 image: {sum(1 for s in class_sizes if s == 1):,}")
    print(f"  Classes with 2 images: {sum(1 for s in class_sizes if s == 2):,}")
    print(f"  Classes with 3+ images: {sum(1 for s in class_sizes if s >= 3):,}")

    # Stratified split
    train_records = []
    val_records = []
    test_records = []

    classes_1_image = 0
    classes_2_images = 0
    classes_3plus_images = 0

    for class_idx, images in classes.items():
        n = len(images)
        random.shuffle(images)

        if n == 1:
            # Only 1 image - put in train only
            train_records.extend(images)
            classes_1_image += 1

        elif n == 2:
            # 2 images - 1 train, 1 val
            train_records.append(images[0])
            val_records.append(images[1])
            classes_2_images += 1

        else:
            # 3+ images - stratified split
            # At least 1 in val, rest distributed 80/10/10
            n_val = max(1, int(n * 0.10))
            n_test = max(1, int(n * 0.10))
            n_train = n - n_val - n_test

            train_records.extend(images[:n_train])
            val_records.extend(images[n_train:n_train + n_val])
            test_records.extend(images[n_train + n_val:])
            classes_3plus_images += 1

    print(f"\nSplit strategy:")
    print(f"  Classes with 1 image (train only): {classes_1_image:,}")
    print(f"  Classes with 2 images (1 train, 1 val): {classes_2_images:,}")
    print(f"  Classes with 3+ images (stratified): {classes_3plus_images:,}")

    print(f"\nFinal splits:")
    print(f"  Train: {len(train_records):,} images")
    print(f"  Val: {len(val_records):,} images")
    print(f"  Test: {len(test_records):,} images")

    # Verify class overlap
    from collections import Counter
    train_classes = set(Counter([r['class_idx'] for r in train_records]).keys())
    val_classes = set(Counter([r['class_idx'] for r in val_records]).keys())

    print(f"\nClass overlap:")
    print(f"  Train classes: {len(train_classes):,}")
    print(f"  Val classes: {len(val_classes):,}")
    print(f"  Overlap: {len(train_classes & val_classes):,}")
    print(f"  Only in train: {len(train_classes - val_classes):,}")
    print(f"  Only in val: {len(val_classes - train_classes):,}")

    # Save
    for name, data in [('train', train_records), ('val', val_records), ('test', test_records)]:
        output_path = Path(f"data/dataset_{name}_stratified.jsonl")
        with open(output_path, 'w') as f:
            for rec in data:
                f.write(json.dumps(rec) + '\n')
        print(f"\n✅ Saved: {output_path}")

if __name__ == "__main__":
    random.seed(42)  # Reproducible
    main()
