#!/usr/bin/env python3
"""
Debug saliency detection to understand why it's failing.
"""

import sys
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.streaming_dataset import LRUImageCache


def debug_saliency_detection():
    """Debug why saliency detection is failing."""

    # Load one cached image
    cache_dir = Path("/media/skanndar/2TB1/aplantida-ml/image_cache")
    cache = LRUImageCache(cache_dir=cache_dir, max_size_gb=220)

    dataset_file = Path(__file__).parent.parent / "data" / "dataset_raw.jsonl"
    with open(dataset_file) as f:
        records = [json.loads(line) for line in f]

    # Find first cached image
    for rec in records:
        cached_path = cache.get(rec['image_url'])
        if cached_path:
            break

    # Load image
    img = Image.open(cached_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = img_np.shape[:2]

    print(f"Image size: {w}x{h}")
    print(f"Plant: {rec.get('latin_name', 'Unknown')}")
    print()

    # Test different saliency methods and thresholds
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Original
    axes[0, 0].imshow(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original")
    axes[0, 0].axis('off')

    # Try Fine-Grained Saliency
    try:
        saliency_fg = cv2.saliency.StaticSaliencyFineGrained_create()
        success, sal_map_fg = saliency_fg.computeSaliency(img_np)

        if success:
            axes[0, 1].imshow(sal_map_fg, cmap='hot')
            axes[0, 1].set_title("Fine-Grained Saliency")
            axes[0, 1].axis('off')

            # Test different thresholds
            for i, thresh in enumerate([0.3, 0.5, 0.7]):
                sal_binary = (sal_map_fg * 255).astype(np.uint8)
                _, binary = cv2.threshold(sal_binary, int(thresh * 255), 255, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    x, y, bw, bh = cv2.boundingRect(largest)

                    img_copy = img_np.copy()
                    cv2.rectangle(img_copy, (x, y), (x+bw, y+bh), (0, 255, 0), 3)

                    axes[1, i].imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
                    axes[1, i].set_title(f"FG thresh={thresh}\nbbox: {bw}x{bh}\n{bw*bh/(w*h)*100:.1f}% of image")
                    axes[1, i].axis('off')
                else:
                    axes[1, i].text(0.5, 0.5, f"No contours\nthresh={thresh}", ha='center', va='center')
                    axes[1, i].axis('off')
        else:
            axes[0, 1].text(0.5, 0.5, "FG Saliency FAILED", ha='center', va='center')
            axes[0, 1].axis('off')

    except Exception as e:
        axes[0, 1].text(0.5, 0.5, f"FG Error:\n{str(e)[:50]}", ha='center', va='center')
        axes[0, 1].axis('off')

    # Try Spectral Residual
    try:
        saliency_sr = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, sal_map_sr = saliency_sr.computeSaliency(img_np)

        if success:
            axes[0, 2].imshow(sal_map_sr, cmap='hot')
            axes[0, 2].set_title("Spectral Residual Saliency")
            axes[0, 2].axis('off')

            # Test with SR
            sal_binary = (sal_map_sr * 255).astype(np.uint8)
            _, binary = cv2.threshold(sal_binary, int(0.5 * 255), 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, bw, bh = cv2.boundingRect(largest)

                img_copy = img_np.copy()
                cv2.rectangle(img_copy, (x, y), (x+bw, y+bh), (0, 255, 0), 3)

                axes[1, 3].imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
                axes[1, 3].set_title(f"SR bbox\n{bw}x{bh}\n{bw*bh/(w*h)*100:.1f}% of image")
                axes[1, 3].axis('off')
            else:
                axes[1, 3].text(0.5, 0.5, "SR: No contours", ha='center', va='center')
                axes[1, 3].axis('off')
        else:
            axes[0, 2].text(0.5, 0.5, "SR Saliency FAILED", ha='center', va='center')
            axes[0, 2].axis('off')

    except Exception as e:
        axes[0, 2].text(0.5, 0.5, f"SR Error:\n{str(e)[:50]}", ha='center', va='center')
        axes[0, 2].axis('off')

    # Simple edge detection as comparison
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    axes[0, 3].imshow(edges, cmap='gray')
    axes[0, 3].set_title("Canny Edges")
    axes[0, 3].axis('off')

    # Show binary thresholds for FG
    if success and 'sal_map_fg' in locals():
        for i, thresh in enumerate([0.2, 0.4, 0.6]):
            sal_binary = (sal_map_fg * 255).astype(np.uint8)
            _, binary = cv2.threshold(sal_binary, int(thresh * 255), 255, cv2.THRESH_BINARY)

            axes[2, i].imshow(binary, cmap='gray')
            axes[2, i].set_title(f"Binary thresh={thresh}")
            axes[2, i].axis('off')

    # Hide unused subplots
    axes[2, 3].axis('off')

    plt.tight_layout()

    output_path = Path(__file__).parent.parent / "results" / "saliency_debug.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')

    print(f"✅ Debug visualization saved to: {output_path}")


if __name__ == "__main__":
    debug_saliency_detection()
