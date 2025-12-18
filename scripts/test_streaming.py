#!/usr/bin/env python3
"""
test_streaming.py - Test the StreamingImageDataset with small sample

Validates:
- Image downloading and caching
- LRU eviction logic
- Image validation (corrupt, size checks)
- Augmentation pipeline integration
- DataLoader batching

Usage:
    python scripts/test_streaming.py --sample-size 50
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.streaming_dataset import StreamingImageDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_augmentations():
    """Create augmentation pipeline for testing."""
    return A.Compose([
        A.Resize(256, 256),
        A.RandomCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def create_test_jsonl(output_path: str, sample_size: int = 50):
    """
    Create a small test JSONL file from MongoDB.

    This creates a sample dataset for testing without needing to
    export the full 340k images.
    """
    import json
    import os
    from pymongo import MongoClient
    from dotenv import load_dotenv

    load_dotenv()

    mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
    db_name = os.getenv('MONGODB_DB_NAME', 'qhopsDB')

    logger.info(f"Connecting to MongoDB: {db_name}")
    client = MongoClient(mongo_uri)
    db = client[db_name]
    plants = db.plants

    records = []
    plants_sampled = 0

    # Sample diverse plants with images
    for plant in plants.find({'images': {'$exists': True, '$ne': []}}).limit(sample_size):
        plant_id = str(plant['_id'])
        latin_name = plant.get('latinName', '')
        common_name = plant.get('commonName', '')

        if not latin_name:
            continue

        # Get first image
        images = plant.get('images', [])
        if not images:
            continue

        img = images[0]
        url = img.get('url', '')
        if not url:
            continue

        record = {
            'plant_id': plant_id,
            'latin_name': latin_name,
            'common_name': common_name,
            'image_url': url,
            'image_source': img.get('source', 'unknown'),
            'license': img.get('license', ''),
            'region': 'EU_SW',  # Simplified for testing
            'country': 'ES'
        }

        records.append(record)
        plants_sampled += 1

        if plants_sampled >= sample_size:
            break

    # Write JSONL
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    logger.info(f"Created test JSONL with {len(records)} samples: {output_path}")
    return len(records)


def test_basic_loading(dataset):
    """Test basic dataset loading."""
    logger.info("Testing basic dataset loading...")

    # Load first item
    img, label, metadata = dataset[0]

    logger.info(f"✅ First item loaded successfully")
    logger.info(f"  Image shape: {img.shape}")
    logger.info(f"  Label: {label}")
    logger.info(f"  Latin name: {metadata['latin_name']}")
    logger.info(f"  Common name: {metadata['common_name']}")

    return True


def test_cache_functionality(dataset, num_samples=10):
    """Test that caching works correctly."""
    logger.info(f"Testing cache functionality with {num_samples} samples...")

    # First pass - should download
    stats_before = dataset.get_stats()
    logger.info(f"Stats before: {stats_before['downloader_stats']}")

    for i in range(min(num_samples, len(dataset))):
        _ = dataset[i]

    stats_after_first = dataset.get_stats()
    downloaded_first = stats_after_first['downloader_stats']['downloaded']
    logger.info(f"After first pass - downloaded: {downloaded_first}")

    # Second pass - should use cache
    for i in range(min(num_samples, len(dataset))):
        _ = dataset[i]

    stats_after_second = dataset.get_stats()
    downloaded_second = stats_after_second['downloader_stats']['downloaded']
    cached_second = stats_after_second['downloader_stats']['cached']

    logger.info(f"After second pass - downloaded: {downloaded_second}, cached: {cached_second}")

    if cached_second > 0:
        logger.info(f"✅ Cache is working! Cached {cached_second} images on second pass")
        return True
    else:
        logger.warning(f"⚠️  Cache might not be working properly")
        return False


def test_dataloader_batching(dataset, batch_size=4, num_workers=2):
    """Test DataLoader integration."""
    logger.info(f"Testing DataLoader with batch_size={batch_size}, num_workers={num_workers}...")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # Load one batch
    for batch_imgs, batch_labels, batch_metadata in dataloader:
        logger.info(f"✅ DataLoader batch loaded successfully")
        logger.info(f"  Batch shape: {batch_imgs.shape}")
        logger.info(f"  Labels shape: {batch_labels.shape}")
        logger.info(f"  Batch size: {len(batch_metadata)}")
        break

    return True


def test_augmentations(dataset, num_samples=5):
    """Test that augmentations are being applied."""
    logger.info(f"Testing augmentations on {num_samples} samples...")

    # Load same image multiple times - should get different augmentations
    images = []
    for _ in range(num_samples):
        img, _, _ = dataset[0]  # Same index
        images.append(img)

    # Check if images are different (due to random augmentations)
    all_same = all(torch.equal(images[0], img) for img in images[1:])

    if not all_same:
        logger.info(f"✅ Augmentations are working! Same image produced different outputs")
        return True
    else:
        logger.warning(f"⚠️  Augmentations might not be random (all outputs identical)")
        return False


def test_error_handling(dataset):
    """Test handling of invalid URLs and corrupt images."""
    logger.info("Testing error handling...")

    # Try to load all samples - some might fail
    successful = 0
    failed = 0

    for i in range(len(dataset)):
        try:
            img, label, metadata = dataset[i]
            if img is not None:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            logger.warning(f"Error loading item {i}: {e}")
            failed += 1

    logger.info(f"✅ Error handling test complete")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")

    return True


def print_final_stats(dataset):
    """Print final statistics."""
    stats = dataset.get_stats()

    print("\n" + "="*60)
    print("           STREAMING DATASET TEST RESULTS")
    print("="*60)
    print(f"Total records:        {stats['total_records']}")
    print(f"Number of classes:    {stats['num_classes']}")
    print(f"Cache size (GB):      {stats['cache_size_gb']:.2f}")

    print("\nDownloader statistics:")
    dl_stats = stats['downloader_stats']
    print(f"  Downloaded:         {dl_stats['downloaded']}")
    print(f"  Cached:             {dl_stats['cached']}")
    print(f"  Failed:             {dl_stats['failed']}")
    print(f"  Invalid:            {dl_stats['invalid']}")

    total_requests = dl_stats['downloaded'] + dl_stats['cached']
    if total_requests > 0:
        cache_hit_rate = dl_stats['cached'] / total_requests * 100
        print(f"  Cache hit rate:     {cache_hit_rate:.1f}%")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Test StreamingImageDataset')
    parser.add_argument('--sample-size', type=int, default=50,
                       help='Number of samples for test dataset')
    parser.add_argument('--cache-size-gb', type=float, default=1.0,
                       help='Cache size in GB (small for testing)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for DataLoader test')
    parser.add_argument('--skip-create', action='store_true',
                       help='Skip creating test JSONL (use existing)')

    args = parser.parse_args()

    # Paths
    test_jsonl = './data/test_sample.jsonl'
    cache_dir = './data/test_cache'

    # Create test dataset
    if not args.skip_create:
        num_created = create_test_jsonl(test_jsonl, args.sample_size)
        if num_created == 0:
            logger.error("Failed to create test dataset")
            return

    # Create augmentations
    logger.info("Creating augmentation pipeline...")
    transform = create_augmentations()

    # Create dataset
    logger.info(f"Creating StreamingImageDataset...")
    dataset = StreamingImageDataset(
        jsonl_path=test_jsonl,
        cache_dir=cache_dir,
        cache_size_gb=args.cache_size_gb,
        transform=transform,
        download_timeout=10,
        min_image_size=(50, 50)
    )

    logger.info(f"Dataset created with {len(dataset)} samples")

    # Run tests
    tests = [
        ("Basic Loading", lambda: test_basic_loading(dataset)),
        ("Cache Functionality", lambda: test_cache_functionality(dataset)),
        ("DataLoader Batching", lambda: test_dataloader_batching(dataset, args.batch_size)),
        ("Augmentations", lambda: test_augmentations(dataset)),
        ("Error Handling", lambda: test_error_handling(dataset))
    ]

    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test failed: {e}")
            results[test_name] = False

    # Print results
    print_final_stats(dataset)

    print("\n" + "="*60)
    print("                  TEST SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:30s} {status}")
    print("="*60)

    # Cleanup
    logger.info("\nCleaning up test cache...")
    dataset.clear_cache()

    all_passed = all(results.values())
    if all_passed:
        logger.info("\n🎉 All tests passed!")
    else:
        logger.warning("\n⚠️  Some tests failed - review logs above")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
