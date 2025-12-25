"""Utility helpers for evaluation/logit scripts."""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root is on sys.path para poder importar `models`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.dataloader_factory import DataLoaderFactory


class TemperatureScaledModel(nn.Module):
    """Wraps a model to apply temperature scaling at inference time."""

    def __init__(self, base_model: nn.Module, temperature: float):
        super().__init__()
        self.base_model = base_model
        self.register_buffer('temperature', torch.tensor([temperature], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base_model(x)
        temp = self.temperature.to(logits.device).clamp(min=1e-6)
        return logits / temp

logger = logging.getLogger(__name__)


def load_class_mapping(
    mapping_path: Optional[str] = None,
    train_jsonl: Optional[str] = None
) -> Dict[str, int]:
    """Load class-to-index mapping from JSON file or derive from a train JSONL."""
    if mapping_path:
        mapping_file = Path(mapping_path)
        if mapping_file.exists():
            with open(mapping_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {str(k): int(v) for k, v in data.items()}

    if not train_jsonl:
        raise ValueError(
            "Provide --class-mapping pointing to an existing file or --train-jsonl to derive it"
        )

    classes = set()
    with open(train_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                classes.add(record['latin_name'])

    mapping = {cls: idx for idx, cls in enumerate(sorted(classes))}
    return mapping


def save_class_mapping(mapping: Dict[str, int], path: Optional[str]):
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved class mapping ({len(mapping)} clases) to {output}")


def build_eval_dataloader(
    config_path: str,
    jsonl_path: str,
    class_mapping: Dict[str, int],
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    mode: str = 'val'
) -> Tuple[DataLoader, dict, dict]:
    """Create a DataLoader for evaluation/logit generation."""
    factory = DataLoaderFactory(config_path)
    dataset = factory.create_dataset(
        jsonl_path=jsonl_path,
        mode=mode,
        class_to_idx=class_mapping
    )

    effective_batch = batch_size or factory.config['training']['batch_size']
    workers = num_workers if num_workers is not None else factory.config['data'].get('num_workers', 4)
    loader = DataLoader(
        dataset,
        batch_size=effective_batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    return loader, dataset, factory.config


def load_model_from_checkpoint(
    config: dict,
    checkpoint_path: str,
    num_classes: int,
    device: torch.device
) -> torch.nn.Module:
    """Instantiate timm model and load weights from checkpoint."""
    model_name = config['model']['name']
    logger.info(f"Creating model {model_name} with {num_classes} clases")
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    missing = model.load_state_dict(state_dict, strict=False)
    if missing and missing.missing_keys:
        logger.warning(f"Missing keys: {missing.missing_keys}")
    if missing and missing.unexpected_keys:
        logger.warning(f"Unexpected keys: {missing.unexpected_keys}")

    model.to(device)

    temperature = checkpoint.get('temperature')
    if temperature is not None:
        logger.info(f"Applying temperature scaling T={float(temperature):.4f}")
        model = TemperatureScaledModel(model, float(temperature))
        model.to(device)

    model.eval()
    return model


def resolve_device(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        dev = torch.device(preferred)
        if dev.type == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA no disponible, usando CPU")
            return torch.device('cpu')
        return dev
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
