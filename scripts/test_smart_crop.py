#!/usr/bin/env python3
"""
Test smart crop on sample images to visualize the difference
between center crop and saliency-based crop.
"""

import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.streaming_dataset import LRUImageCache, ImageDownloader
from models.smart_crop import SaliencyCropper


def test_smart_crop(num_samples=6):
    """Test smart crop on random cached images."""

    print("=" * 70)
    print("  Smart Crop Visualization Test")
    print("=" * 70)
    print()

    # Initialize
    cache_dir = Path("/media/skanndar/2TB1/aplantida-ml/image_cache")
    cache = LRUImageCache(cache_dir=cache_dir, max_size_gb=220)
    cropper = SaliencyCropper(target_size=384)

    # Load dataset
    dataset_file = Path(__file__).parent.parent / "data" / "dataset_raw.jsonl"
    with open(dataset_file) as f:
        records = [json.loads(line) for line in f]

    print(f"Loaded {len(records):,} records")
    print()

    # Find cached images
    cached_records = []
    for rec in records:
        if cache.get(rec['image_url']):
            cached_records.append(rec)
        if len(cached_records) >= num_samples * 2:  # Get extra in case some fail
            break

    print(f"Found {len(cached_records)} cached images for testing")
    print()

    # Test on samples
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    successful_samples = 0

    for i, rec in enumerate(cached_records):
        if successful_samples >= num_samples:
            break

        url = rec['image_url']
        plant_name = rec.get('latin_name', 'Unknown')
        cached_path = cache.get(url)

        if not cached_path:
            continue

        try:
            # Load image
            img = Image.open(cached_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            print(f"[{successful_samples + 1}/{num_samples}] {plant_name}")
            print(f"   Original size: {img.size}")

            # Get saliency bbox
            import cv2
            img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            bbox = cropper.get_saliency_bbox(img_np)

            # Original image with bbox overlay
            axes[successful_samples, 0].imshow(img)
            axes[successful_samples, 0].set_title(f"Original\n{plant_name[:30]}\n{img.size[0]}x{img.size[1]}px")
            axes[successful_samples, 0].axis('off')

            if bbox:
                x, y, w, h = bbox
                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=3,
                    edgecolor='red',
                    facecolor='none'
                )
                axes[successful_samples, 0].add_patch(rect)
                print(f"   Saliency bbox: ({x}, {y}, {w}, {h})")

            # Center crop
            center_cropped = cropper.center_crop(img)
            axes[successful_samples, 1].imshow(center_cropped)
            axes[successful_samples, 1].set_title(f"Center Crop\n384x384px")
            axes[successful_samples, 1].axis('off')

            # Smart crop
            smart_cropped = cropper.smart_crop(img)
            axes[successful_samples, 2].imshow(smart_cropped)
            crop_method = "Smart Crop" if bbox else "Center Crop (fallback)"
            axes[successful_samples, 2].set_title(f"{crop_method}\n384x384px")
            axes[successful_samples, 2].axis('off')

            successful_samples += 1
            print()

        except Exception as e:
            print(f"   Error: {e}")
            print()
            continue

    # Remove unused subplots
    if successful_samples < num_samples:
        for i in range(successful_samples, num_samples):
            for j in range(3):
                fig.delaxes(axes[i, j])

    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent.parent / "results" / "smart_crop_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_path}")
    print()
    print(f"Successfully tested {successful_samples} images")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test smart crop visualization")
    parser.add_argument("--samples", type=int, default=6, help="Number of samples to test")
    args = parser.parse_args()

    test_smart_crop(num_samples=args.samples)
