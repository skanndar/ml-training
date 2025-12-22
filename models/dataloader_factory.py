"""
dataloader_factory.py - DataLoader Factory with Configuration

Creates PyTorch DataLoaders from YAML configs with:
- Streaming dataset integration
- Augmentation pipelines
- Regional filtering
- License filtering
- Multi-worker support

Usage:
    from models.dataloader_factory import create_dataloaders

    train_loader, val_loader = create_dataloaders(
        config_path='./config/teacher_global.yaml'
    )
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import albumentations as A
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from models.streaming_dataset import StreamingImageDataset

logger = logging.getLogger(__name__)


class DataLoaderFactory:
    """Factory for creating DataLoaders from configuration."""

    def __init__(self, config_path: str, paths_config: str = './config/paths.yaml'):
        """
        Args:
            config_path: Path to model config (teacher_global.yaml, etc.)
            paths_config: Path to paths configuration
        """
        self.config_path = Path(config_path)
        self.paths_config = Path(paths_config)

        # Load configs
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        with open(self.paths_config, 'r') as f:
            self.paths = yaml.safe_load(f)

        logger.info(f"Loaded config from {config_path}")

    def create_augmentations(self, mode: str = 'train') -> A.Compose:
        """
        Create augmentation pipeline from config.

        Args:
            mode: 'train' or 'val'

        Returns:
            Albumentations Compose pipeline
        """
        aug_config = self.config['augmentation'][mode]

        transforms = []

        # Resize
        if 'resize' in aug_config:
            size = aug_config['resize']
            transforms.append(A.Resize(size, size))

        # Crop
        if mode == 'train' and 'crop' in aug_config:
            crop_size = aug_config['crop']
            transforms.append(A.RandomCrop(crop_size, crop_size))
        elif mode == 'val' and 'center_crop' in aug_config:
            crop_size = aug_config['center_crop']
            transforms.append(A.CenterCrop(crop_size, crop_size))

        # Training-specific augmentations
        if mode == 'train':
            if 'horizontal_flip' in aug_config:
                transforms.append(A.HorizontalFlip(p=aug_config['horizontal_flip']))

            if 'vertical_flip' in aug_config:
                transforms.append(A.VerticalFlip(p=aug_config['vertical_flip']))

            if 'rotation' in aug_config:
                transforms.append(A.Rotate(limit=aug_config['rotation'], p=0.3))

            if 'color_jitter' in aug_config:
                jitter = aug_config['color_jitter']
                transforms.append(A.ColorJitter(
                    brightness=jitter.get('brightness', 0.2),
                    contrast=jitter.get('contrast', 0.2),
                    saturation=jitter.get('saturation', 0.2),
                    hue=jitter.get('hue', 0.1),
                    p=0.5
                ))

            if 'blur' in aug_config:
                transforms.append(A.Blur(blur_limit=3, p=aug_config['blur']))

        # Normalization (always last before ToTensor)
        if 'normalize' in aug_config:
            norm = aug_config['normalize']
            transforms.append(A.Normalize(
                mean=norm['mean'],
                std=norm['std']
            ))

        # Convert to tensor
        transforms.append(ToTensorV2())

        return A.Compose(transforms)

    def create_dataset(
        self,
        jsonl_path: str,
        mode: str = 'train',
        region_filter: Optional[str] = None,
        class_to_idx: Optional[Dict] = None
    ) -> StreamingImageDataset:
        """
        Create StreamingImageDataset.

        Args:
            jsonl_path: Path to JSONL file
            mode: 'train' or 'val'
            region_filter: Optional region filter (EU_SW, etc.)
            class_to_idx: Optional pre-built class mapping (for val to use train's mapping)

        Returns:
            StreamingImageDataset instance
        """
        # Get augmentations
        transform = self.create_augmentations(mode)

        # Get cache config
        cache_dir = self.paths['data']['cache_dir']
        cache_size_gb = self.config.get('data', {}).get('cache_size_gb', 100)

        # Get smart crop config
        smart_crop = self.config.get('data', {}).get('smart_crop', False)
        image_size = self.config.get('data', {}).get('image_size', 224)

        # Create dataset
        dataset = StreamingImageDataset(
            jsonl_path=jsonl_path,
            cache_dir=cache_dir,
            cache_size_gb=cache_size_gb,
            transform=transform,
            download_timeout=10,
            min_image_size=(50, 50),
            smart_crop=smart_crop,
            smart_crop_size=image_size,
            class_to_idx=class_to_idx
        )

        # Apply region filter if specified
        if region_filter:
            logger.info(f"Applying region filter: {region_filter}")
            original_count = len(dataset.records)
            dataset.records = [
                r for r in dataset.records
                if r.get('region') == region_filter
            ]
            logger.info(f"Filtered from {original_count} to {len(dataset.records)} records")

            # Rebuild class mapping after filtering
            dataset.class_to_idx = dataset._build_class_mapping()
            logger.info(f"Classes after filtering: {len(dataset.class_to_idx)}")

        return dataset

    def create_dataloader(
        self,
        dataset: StreamingImageDataset,
        mode: str = 'train'
    ) -> DataLoader:
        """
        Create DataLoader from dataset.

        Args:
            dataset: StreamingImageDataset instance
            mode: 'train' or 'val'

        Returns:
            PyTorch DataLoader
        """
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['data'].get('num_workers', 4)

        # Training-specific settings
        shuffle = (mode == 'train')
        drop_last = (mode == 'train')

        # Pin memory if CUDA available
        pin_memory = torch.cuda.is_available()

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=(num_workers > 0)
        )

        return dataloader

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, Dict]:
        """
        Create train and validation DataLoaders.

        Returns:
            Tuple of (train_loader, val_loader, info_dict)
        """
        # Get paths from config
        train_jsonl = self.config['data']['train_jsonl']
        val_jsonl = self.config['data']['val_jsonl']
        region_filter = self.config['data'].get('region_filter', None)

        logger.info(f"Creating datasets...")
        logger.info(f"  Train: {train_jsonl}")
        logger.info(f"  Val: {val_jsonl}")
        if region_filter:
            logger.info(f"  Region filter: {region_filter}")

        # Create datasets
        # CRITICAL: Train creates its class mapping first
        train_dataset = self.create_dataset(
            train_jsonl,
            mode='train',
            region_filter=region_filter
        )

        # CRITICAL: Val MUST use train's class mapping to avoid index mismatch
        val_dataset = self.create_dataset(
            val_jsonl,
            mode='val',
            region_filter=None,  # Usually validate on full dataset
            class_to_idx=train_dataset.class_to_idx  # Use train's mapping!
        )

        # Create dataloaders
        train_loader = self.create_dataloader(train_dataset, mode='train')
        val_loader = self.create_dataloader(val_dataset, mode='val')

        # Info dict
        info = {
            'num_classes': len(train_dataset.class_to_idx),
            'class_to_idx': train_dataset.class_to_idx,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'batch_size': self.config['training']['batch_size']
        }

        logger.info(f"DataLoaders created:")
        logger.info(f"  Train: {info['train_size']} samples, {len(train_loader)} batches")
        logger.info(f"  Val: {info['val_size']} samples, {len(val_loader)} batches")
        logger.info(f"  Classes: {info['num_classes']}")

        return train_loader, val_loader, info


def create_dataloaders(
    config_path: str,
    paths_config: str = './config/paths.yaml'
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Convenience function to create DataLoaders.

    Args:
        config_path: Path to model config (teacher_global.yaml, etc.)
        paths_config: Path to paths configuration

    Returns:
        Tuple of (train_loader, val_loader, info_dict)

    Example:
        >>> train_loader, val_loader, info = create_dataloaders(
        ...     config_path='./config/teacher_global.yaml'
        ... )
        >>> print(f"Training on {info['num_classes']} classes")
        >>> for images, labels, metadata in train_loader:
        ...     # Training loop
        ...     pass
    """
    factory = DataLoaderFactory(config_path, paths_config)
    return factory.create_dataloaders()


# Example usage
if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)

    # Test with teacher_global config
    train_loader, val_loader, info = create_dataloaders(
        config_path='./config/teacher_global.yaml'
    )

    print(f"\nDataLoader Info:")
    print(f"  Classes: {info['num_classes']}")
    print(f"  Train samples: {info['train_size']}")
    print(f"  Val samples: {info['val_size']}")
    print(f"  Batch size: {info['batch_size']}")

    # Test loading one batch
    print(f"\nLoading first batch...")
    for batch_images, batch_labels, batch_metadata in train_loader:
        print(f"  Batch shape: {batch_images.shape}")
        print(f"  Labels shape: {batch_labels.shape}")
        print(f"  Sample metadata keys: {list(batch_metadata.keys())}")
        break
