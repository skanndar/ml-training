#!/usr/bin/env python3
"""
Test the smart rate limiting system by:
1. Loading dataset_raw.jsonl (741k images)
2. Finding images NOT in cache
3. Attempting to download them
4. Verifying rate limiting kicks in when 429 detected
"""

import hashlib
import json
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.streaming_dataset import LRUImageCache, ImageDownloader

def url_to_filename(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest() + ".jpg"

def main():
    print("=" * 70)
    print("  Testing Smart Rate Limiting System")
    print("=" * 70)
    print()

    # Initialize cache and downloader
    cache_dir = Path("/media/skanndar/2TB1/aplantida-ml/image_cache")
    cache = LRUImageCache(cache_dir=cache_dir, max_size_gb=220)
    downloader = ImageDownloader(timeout=10, max_retries=3)

    print(f"Cache size: {cache.current_size_bytes / 1024**3:.2f}GB")
    print(f"Cached files: {len(cache.files):,}")
    print()

    # Load dataset
    dataset_file = Path("data/dataset_raw.jsonl")
    records = []
    with open(dataset_file) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Loaded {len(records):,} records from dataset_raw.jsonl")
    print()

    # Find uncached images
    uncached = []
    for rec in records:
        url = rec['image_url']
        if not cache.get(url):
            uncached.append(rec)

    print(f"Uncached images: {len(uncached):,}/{len(records):,} ({len(uncached)/len(records)*100:.1f}%)")
    print()

    if len(uncached) == 0:
        print("✅ All images are already cached! No downloads needed.")
        return

    # Test downloading first 50 uncached images
    test_count = min(50, len(uncached))
    print(f"Testing download of first {test_count} uncached images...")
    print()

    successful = 0
    failed = 0
    rate_limited = 0
    start_time = time.time()

    for i, rec in enumerate(uncached[:test_count], 1):
        url = rec['image_url']
        plant_name = rec.get('latin_name', 'Unknown')

        print(f"[{i}/{test_count}] {plant_name[:40]:<40} ", end='', flush=True)

        # Attempt download
        image_bytes = downloader.download(url)

        if image_bytes:
            # Validate
            img = downloader.validate(image_bytes)
            if img:
                # Cache it
                cache.put(url, image_bytes)
                successful += 1
                print(f"✅ Downloaded ({len(image_bytes)//1024}KB)")
            else:
                failed += 1
                print(f"❌ Invalid")
        else:
            # Check if rate limited
            if downloader.rate_limited:
                rate_limited += 1
                backoff_remaining = int(downloader.rate_limit_until - time.time())
                print(f"⏸️  Rate limited (backoff: {backoff_remaining}s remaining)")
            else:
                failed += 1
                print(f"❌ Failed")

        # Small delay between requests
        time.sleep(0.1)

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("  Test Results")
    print("=" * 70)
    print(f"Total tested: {test_count}")
    print(f"Successful: {successful} ({successful/test_count*100:.1f}%)")
    print(f"Failed: {failed} ({failed/test_count*100:.1f}%)")
    print(f"Rate limited: {rate_limited} ({rate_limited/test_count*100:.1f}%)")
    print(f"Time elapsed: {elapsed:.1f}s")
    print()

    print("Downloader stats:")
    print(f"  Downloaded: {downloader.stats['downloaded']}")
    print(f"  Failed: {downloader.stats['failed']}")
    print(f"  Invalid: {downloader.stats['invalid']}")
    print(f"  Rate limited skips: {downloader.stats['rate_limited_skips']}")
    print()

    if downloader.rate_limited:
        backoff_remaining = int(downloader.rate_limit_until - time.time())
        print(f"⚠️  Currently rate limited!")
        print(f"   Backoff period: {downloader.rate_limit_backoff}s")
        print(f"   Time remaining: {backoff_remaining}s")
        print(f"   Consecutive 429s: {downloader.consecutive_429s}")
        print()
        print("✅ Smart rate limiting is WORKING - downloads paused automatically")
    else:
        print("✅ No rate limiting detected - downloads proceeding normally")

    print()
    print(f"Cache size after test: {cache.current_size_bytes / 1024**3:.2f}GB")
    print(f"Cached files: {len(cache.files):,}")

if __name__ == "__main__":
    main()
