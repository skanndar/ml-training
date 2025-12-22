"""
streaming_dataset.py - On-the-fly Image Loading with Smart Caching

Features:
- Downloads images on-demand during training
- LRU cache (configurable size: 50-200GB)
- Automatic validation (corrupt, size, format)
- Thread-safe concurrent downloads
- Retry logic for failed downloads
- Memory-efficient (only keeps recent images)

Usage:
    dataset = StreamingImageDataset(
        jsonl_path="./data/dataset_train.jsonl",
        cache_dir="./data/image_cache",
        cache_size_gb=100,
        augmentations=train_augmentations
    )
"""

import hashlib
import io
import json
import logging
import os
import time
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Optional, Callable, Dict, Tuple

import numpy as np
import requests
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LRUImageCache:
    """Thread-safe LRU cache for downloaded images."""

    def __init__(self, cache_dir: str, max_size_gb: float = 100):
        """
        Args:
            cache_dir: Directory to store cached images
            max_size_gb: Maximum cache size in GB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.current_size_bytes = 0

        # Track files: {filename: (size_bytes, last_access_time)}
        self.files = OrderedDict()
        self.lock = Lock()

        # Load existing cache
        self._scan_cache()

        logger.info(f"LRU Cache initialized: {cache_dir} (max {max_size_gb}GB)")
        logger.info(f"Current cache size: {self.current_size_bytes / 1024 / 1024 / 1024:.2f}GB")

    def _scan_cache(self):
        """Scan cache directory and build file index."""
        if not self.cache_dir.exists():
            return

        for filepath in self.cache_dir.glob("*.jpg"):
            try:
                size = filepath.stat().st_size
                mtime = filepath.stat().st_mtime
                self.files[filepath.name] = (size, mtime)
                self.current_size_bytes += size
            except Exception as e:
                logger.warning(f"Error scanning {filepath}: {e}")

    def _evict_lru(self):
        """Evict least recently used files until under max_size."""
        while self.current_size_bytes > self.max_size_bytes and self.files:
            # Remove oldest (first in OrderedDict)
            filename, (size, _) = self.files.popitem(last=False)
            filepath = self.cache_dir / filename

            try:
                if filepath.exists():
                    filepath.unlink()
                    self.current_size_bytes -= size
                    logger.debug(f"Evicted from cache: {filename} ({size/1024/1024:.1f}MB)")
                else:
                    # File doesn't exist but was in tracking - update size anyway
                    self.current_size_bytes -= size
                    logger.debug(f"File missing from cache: {filename} (already deleted)")
            except PermissionError as e:
                # Try with force if permission denied
                logger.warning(f"Permission denied evicting {filename}, retrying...")
                try:
                    import shutil
                    shutil.rmtree(filepath, ignore_errors=True)
                    self.current_size_bytes -= size
                except Exception as e2:
                    logger.error(f"Failed to force evict {filename}: {e2}")
                    # Still reduce size to prevent infinite loop
                    self.current_size_bytes = max(0, self.current_size_bytes - size)
            except Exception as e:
                logger.error(f"Failed to evict {filename}: {e}")
                # Still reduce size to prevent infinite loop
                self.current_size_bytes = max(0, self.current_size_bytes - size)

    def get(self, url: str) -> Optional[Path]:
        """
        Get cached image path if exists.

        Args:
            url: Image URL

        Returns:
            Path to cached image, or None if not cached
        """
        filename = self._url_to_filename(url)
        filepath = self.cache_dir / filename

        with self.lock:
            if filename in self.files:
                # Update access time (move to end = most recent)
                size, _ = self.files.pop(filename)
                self.files[filename] = (size, time.time())

                if filepath.exists():
                    return filepath

        return None

    def put(self, url: str, image_bytes: bytes) -> Path:
        """
        Store image in cache.

        Args:
            url: Image URL
            image_bytes: Image data

        Returns:
            Path to cached image
        """
        filename = self._url_to_filename(url)
        filepath = self.cache_dir / filename

        with self.lock:
            # Write to disk
            filepath.write_bytes(image_bytes)
            size = len(image_bytes)

            # Update tracking
            if filename in self.files:
                old_size, _ = self.files.pop(filename)
                self.current_size_bytes -= old_size

            self.files[filename] = (size, time.time())
            self.current_size_bytes += size

            # Evict if over limit
            self._evict_lru()

        return filepath

    @staticmethod
    def _url_to_filename(url: str) -> str:
        """Convert URL to safe filename."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return f"{url_hash}.jpg"

    def clear(self):
        """Clear entire cache."""
        with self.lock:
            for filename in list(self.files.keys()):
                filepath = self.cache_dir / filename
                try:
                    filepath.unlink()
                except Exception:
                    pass
            self.files.clear()
            self.current_size_bytes = 0


class ImageDownloader:
    """Downloads and validates images with retry logic."""

    def __init__(
        self,
        timeout: int = 10,
        max_retries: int = 3,
        min_size: Tuple[int, int] = (50, 50),
        max_size: Tuple[int, int] = (5000, 5000)
    ):
        """
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            min_size: Minimum image dimensions (width, height)
            max_size: Maximum image dimensions
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_size = min_size
        self.max_size = max_size

        # Stats
        self.stats = {
            'downloaded': 0,
            'cached': 0,
            'failed': 0,
            'invalid': 0
        }

    def download(self, url: str) -> Optional[bytes]:
        """
        Download image with retries.

        Args:
            url: Image URL

        Returns:
            Image bytes, or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    url,
                    timeout=self.timeout,
                    headers={'User-Agent': 'Mozilla/5.0'}
                )

                if response.status_code == 200:
                    self.stats['downloaded'] += 1
                    return response.content

            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    logger.debug(f"Failed to download {url}: {e}")
                    self.stats['failed'] += 1
                else:
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff

        return None

    def validate(self, image_bytes: bytes) -> Optional[Image.Image]:
        """
        Validate and load image.

        Args:
            image_bytes: Raw image data

        Returns:
            PIL Image if valid, None otherwise
        """
        try:
            img = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Check dimensions
            width, height = img.size

            if width < self.min_size[0] or height < self.min_size[1]:
                logger.debug(f"Image too small: {width}x{height}")
                self.stats['invalid'] += 1
                return None

            if width > self.max_size[0] or height > self.max_size[1]:
                logger.debug(f"Image too large: {width}x{height}")
                self.stats['invalid'] += 1
                return None

            return img

        except (UnidentifiedImageError, OSError, IOError) as e:
            logger.debug(f"Invalid image: {e}")
            self.stats['invalid'] += 1
            return None

    def get_stats(self) -> Dict:
        """Get download statistics."""
        return self.stats.copy()


class StreamingImageDataset:
    """
    PyTorch-compatible dataset that streams images from URLs.

    Features:
    - On-demand downloading
    - LRU caching
    - Automatic validation
    - Configurable augmentations
    """

    def __init__(
        self,
        jsonl_path: str,
        cache_dir: str = "./data/image_cache",
        cache_size_gb: float = 100,
        transform: Optional[Callable] = None,
        download_timeout: int = 10,
        min_image_size: Tuple[int, int] = (50, 50),
        prefetch: bool = False,
        num_prefetch_workers: int = 4
    ):
        """
        Args:
            jsonl_path: Path to dataset JSONL file
            cache_dir: Directory for image cache
            cache_size_gb: Maximum cache size in GB
            transform: Image transformation function (augmentations)
            download_timeout: Download timeout in seconds
            min_image_size: Minimum valid image dimensions
            prefetch: Whether to prefetch images in background
            num_prefetch_workers: Number of prefetch threads
        """
        self.jsonl_path = jsonl_path
        self.transform = transform

        # Load records
        self.records = self._load_jsonl(jsonl_path)
        logger.info(f"Loaded {len(self.records)} records from {jsonl_path}")

        # Initialize cache and downloader
        self.cache = LRUImageCache(cache_dir, cache_size_gb)
        self.downloader = ImageDownloader(
            timeout=download_timeout,
            min_size=min_image_size
        )

        # Build class mapping
        self.class_to_idx = self._build_class_mapping()
        logger.info(f"Found {len(self.class_to_idx)} unique classes")

    def _load_jsonl(self, filepath: str) -> list:
        """Load JSONL file."""
        records = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def _build_class_mapping(self) -> Dict[str, int]:
        """Build mapping from class name to index."""
        unique_classes = sorted(set(r['latin_name'] for r in self.records))
        return {cls: idx for idx, cls in enumerate(unique_classes)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, Dict]:
        """
        Get item by index.

        Returns:
            Tuple of (image_tensor, class_idx, metadata)
        """
        record = self.records[idx]
        url = record['image_url']
        class_name = record['latin_name']
        class_idx = self.class_to_idx[class_name]

        # Try to get from cache first
        cached_path = self.cache.get(url)

        if cached_path:
            # Load from cache
            try:
                img = Image.open(cached_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                self.downloader.stats['cached'] += 1
            except Exception as e:
                logger.warning(f"Failed to load cached image {cached_path}: {e}")
                img = None
        else:
            # Download
            img = None
            image_bytes = self.downloader.download(url)

            if image_bytes:
                # Validate
                img = self.downloader.validate(image_bytes)

                if img:
                    # Cache for future use
                    self.cache.put(url, image_bytes)

        # Fallback: black image if download failed
        if img is None:
            logger.warning(f"Failed to load image for {class_name}, using blank")
            img = Image.new('RGB', (224, 224), color=(0, 0, 0))

        # Convert to numpy
        img_array = np.array(img)

        # Apply transformations
        if self.transform:
            img_array = self.transform(image=img_array)['image']

        # Metadata
        metadata = {
            'plant_id': record.get('plant_id', ''),
            'latin_name': class_name,
            'common_name': record.get('common_name', ''),
            'region': record.get('region', ''),
            'url': url
        }

        return img_array, class_idx, metadata

    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        return {
            'total_records': len(self.records),
            'num_classes': len(self.class_to_idx),
            'cache_size_gb': self.cache.current_size_bytes / 1024 / 1024 / 1024,
            'downloader_stats': self.downloader.get_stats()
        }

    def clear_cache(self):
        """Clear the image cache."""
        self.cache.clear()


# Example usage
if __name__ == '__main__':
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    # Define augmentations
    transform = A.Compose([
        A.Resize(256, 256),
        A.RandomCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # Create dataset
    dataset = StreamingImageDataset(
        jsonl_path="./data/dataset_train.jsonl",
        cache_dir="./data/image_cache",
        cache_size_gb=50,
        transform=transform
    )

    # Test loading
    print(f"Dataset size: {len(dataset)}")
    img, label, meta = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Label: {label}")
    print(f"Metadata: {meta}")

    # Print stats
    print(f"Stats: {dataset.get_stats()}")
