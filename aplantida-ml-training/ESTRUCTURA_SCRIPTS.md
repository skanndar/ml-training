# Estructura de Repo y Scripts

## 1. Organización del Proyecto

```
~/ml-training/                          # Raíz
├── README.md                           # Este documento
├── requirements.txt                    # Python dependencies
├── environment.yaml                    # Conda environment
├── pyproject.toml                      # Python packaging
│
├── config/                             # Configuración
│   ├── teacher_global.yaml
│   ├── teacher_regional.yaml
│   ├── student.yaml
│   ├── distillation.yaml
│   └── paths.yaml                      # Rutas globales
│
├── scripts/                            # Ejecutables
│   ├── __init__.py
│   ├── audit_dataset.py
│   ├── export_dataset.py
│   ├── split_dataset.py
│   ├── data_loader.py
│   ├── augmentations.py
│   ├── train_teacher.py
│   ├── eval_teacher.py
│   ├── compute_teacher_logits.py
│   ├── combine_teacher_logits.py
│   ├── train_student_distill.py
│   ├── train_student_finetune.py
│   ├── calibrate_model.py
│   ├── temperature_scaling.py
│   ├── find_confidence_threshold.py
│   ├── quantize_model.py
│   ├── export_to_savedmodel.py
│   ├── export_to_tfjs.py
│   ├── validate_tfjs_export.py
│   ├── eval_metrics.py
│   ├── eval_by_region.py
│   ├── analyze_confusions.py
│   ├── generate_manifest.py
│   ├── license_checker.py
│   ├── dashboard.py
│   ├── reproduce_full_pipeline.sh
│   └── cleanup.sh
│
├── models/                             # Código de modelos
│   ├── __init__.py
│   ├── student_architecture.py
│   ├── distillation_loss.py
│   ├── teacher_ensemble.py
│   └── metrics.py
│
├── data/                               # Dataset (NO commitear)
│   ├── dataset_raw.jsonl
│   ├── dataset_splits.jsonl
│   ├── dataset_eu_sw_train.jsonl
│   ├── soft_labels_combined_train.npz
│   ├── teacher_global_logits_train.npz
│   ├── teacher_regional_logits_train.npz
│   ├── data_cache/
│   │   ├── plant_id_1_abc123.jpg
│   │   ├── plant_id_2_def456.jpg
│   │   └── ...
│   └── .gitignore                      # Ignorar imágenes
│
├── checkpoints/                        # Modelos durante entrenamiento
│   ├── teacher_global/
│   │   ├── epoch_01/
│   │   ├── epoch_02/
│   │   └── best_model.pt
│   ├── teacher_regional/
│   │   └── ...
│   ├── student_distill/
│   │   └── ...
│   └── student_finetune/
│       └── best_model.pt
│
├── results/                            # Outputs finales
│   ├── baseline_v0/
│   │   └── metrics.json
│   ├── teacher_global_v1/
│   │   ├── best_model.pt
│   │   ├── eval_test.json
│   │   └── full_evaluation.json
│   ├── teacher_regional_v1/
│   │   ├── best_model.pt
│   │   └── eval_by_region.json
│   ├── student_distill_v1/
│   │   └── ...
│   ├── student_finetune_v1/
│   │   ├── best_model.pt
│   │   ├── calibration.json
│   │   ├── model_fp16.pt
│   │   └── full_evaluation.json
│   ├── TRAINING_MANIFEST_v1.0.yaml
│   └── eval_dashboard.png
│
├── dist/                               # TF.js export
│   └── models/
│       └── student_v1.0/
│           ├── model.json
│           ├── group1-shard1of3.bin
│           ├── group1-shard2of3.bin
│           └── group1-shard3of3.bin
│
├── docs/                               # Documentación
│   ├── TRAINING.md                     # Notas de entrenamiento
│   └── RESULTS.md                      # Resultados finales
│
├── tests/                              # Tests
│   ├── test_dataloader.py
│   ├── test_models.py
│   ├── test_distillation.py
│   └── test_export.py
│
└── .gitignore                          # Ignorar archivos grandes
```

---

## 2. Requirements & Environment

### requirements.txt

```
torch==2.1.0
torchvision==0.16.0
tensorflow==2.14.0
tensorflowjs==4.11.0

pymongo==4.6.0
requests==2.31.0

scikit-learn==1.3.2
pandas==2.1.3
numpy==1.26.2
matplotlib==3.8.2
seaborn==0.13.0
albumentations==1.3.1

tqdm==4.66.1
pyyaml==6.0.1
python-dotenv==1.0.0
```

### environment.yaml (Conda)

```yaml
name: aplantida-ml
channels:
  - pytorch
  - conda-forge
dependencies:
  - python=3.11
  - pytorch::pytorch::2.1.0 [py311_cuda11.8*]
  - pytorch::pytorch-cuda=11.8
  - pytorch::torchvision=0.16.0
  - pytorch::pytorch-lightning
  - conda-forge::tensorflow=2.14.0
  - conda-forge::tensorflowjs=4.11.0
  - conda-forge::mongodb=6.0
  - conda-forge::mongosh=1.10.0
  - pip
  - pip::scikit-learn
  - pip::tensorflowjs
```

### Setup

```bash
# Opción 1: Conda
conda env create -f environment.yaml
conda activate aplantida-ml

# Opción 2: Venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o: venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

---

## 3. Archivos de Configuración

### config/paths.yaml

```yaml
# Rutas globales
data:
  mongo_uri: "mongodb://localhost:27017"
  db_name: "aplantida_db"
  cache_dir: "./data/data_cache"

datasets:
  raw_jsonl: "./data/dataset_raw.jsonl"
  splits_jsonl: "./data/dataset_splits.jsonl"
  eu_sw_jsonl: "./data/dataset_eu_sw_train.jsonl"
  soft_labels: "./data/soft_labels_combined_train.npz"

models:
  checkpoints_dir: "./checkpoints"
  results_dir: "./results"
  dist_dir: "./dist/models"

logging:
  log_dir: "./logs"
  level: "INFO"
```

### config/teacher_global.yaml

```yaml
model:
  name: "vit_base_patch16_224"
  pretrained: true
  num_classes: 9000

training:
  learning_rate: 1e-5
  batch_size: 64
  epochs: 15
  warmup_epochs: 2
  optimizer: "adamw"
  weight_decay: 0.01
  gradient_clip: 1.0

data:
  train_split: "train"
  image_size: 224
  augmentation: "aggressive"

regularization:
  dropout: 0.2
  mixup_alpha: 0.8
  cutmix_alpha: 0.8

callbacks:
  early_stopping: true
  early_stopping_patience: 3
  save_checkpoint_every: 1  # epochs
  save_best: true

device:
  type: "cuda"  # o "cpu"
  mixed_precision: true
```

### config/student.yaml

```yaml
model:
  name: "mobilenetv2"
  alpha: 1.0
  num_classes: 9000
  dropout_rate: 0.2

distillation:
  temperature: 3.0
  alpha: 0.7  # KL weight

teachers:
  - path: "./results/teacher_global_v1/best_model.pt"
    weight: 0.5
  - path: "./results/teacher_regional_v1/best_model.pt"
    weight: 0.5

training:
  phase1:
    name: "distillation"
    epochs: 10
    learning_rate: 1e-4
    batch_size: 128
  phase2:
    name: "finetuning"
    epochs: 10
    learning_rate: 5e-5
    batch_size: 128

callbacks:
  early_stopping_patience: 5
  save_best: true

device:
  type: "cuda"
  mixed_precision: true
```

---

## 4. Script Principal Reproducible

### reproduce_full_pipeline.sh

```bash
#!/bin/bash
set -e  # Exit on error

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Aplantida ML Training Pipeline ===${NC}"

# Configuración
SEED=42
DATA_DIR="./data"
RESULTS_DIR="./results"
CHECKPOINTS_DIR="./checkpoints"

# Crear directorios
mkdir -p $DATA_DIR $RESULTS_DIR $CHECKPOINTS_DIR

# ============ FASE 0: Auditoría ============
echo -e "${BLUE}[Fase 0] Auditing dataset...${NC}"
python scripts/audit_dataset.py \
  --log_level INFO

echo -e "${GREEN}✓ Dataset audited${NC}\n"

# ============ FASE 1: Export ============
echo -e "${BLUE}[Fase 1] Exporting from MongoDB...${NC}"
python scripts/export_dataset.py \
  --output_dir $DATA_DIR \
  --min_images_per_species 5 \
  --log_level INFO

echo -e "${BLUE}Splitting train/val/test...${NC}"
python scripts/split_dataset.py \
  --jsonl_path $DATA_DIR/dataset_raw.jsonl \
  --output_dir $DATA_DIR \
  --seed $SEED

echo -e "${GREEN}✓ Data exported and split${NC}\n"

# ============ FASE 1: Teacher Global ============
echo -e "${BLUE}[Fase 1] Training teacher global...${NC}"
python scripts/train_teacher.py \
  --config config/teacher_global.yaml \
  --output_dir $RESULTS_DIR/teacher_global_v1 \
  --seed $SEED \
  --log_level INFO

echo -e "${BLUE}Evaluating teacher global...${NC}"
python scripts/eval_teacher.py \
  --model $RESULTS_DIR/teacher_global_v1/best_model.pt \
  --test_jsonl $DATA_DIR/dataset_splits.jsonl \
  --output $RESULTS_DIR/teacher_global_v1/eval_test.json

echo -e "${BLUE}Computing teacher global logits...${NC}"
python scripts/compute_teacher_logits.py \
  --model $RESULTS_DIR/teacher_global_v1/best_model.pt \
  --train_jsonl $DATA_DIR/dataset_splits.jsonl \
  --output $DATA_DIR/teacher_global_logits_train.npz

echo -e "${GREEN}✓ Teacher global ready${NC}\n"

# ============ FASE 2: Teacher Regional ============
echo -e "${BLUE}[Fase 2] Preparing regional dataset...${NC}"
python scripts/prepare_regional_dataset.py \
  --input_jsonl $DATA_DIR/dataset_splits.jsonl \
  --output_dir $DATA_DIR \
  --region EU_SW

echo -e "${BLUE}Training teacher regional...${NC}"
python scripts/train_teacher.py \
  --config config/teacher_regional.yaml \
  --output_dir $RESULTS_DIR/teacher_regional_v1 \
  --seed $SEED

echo -e "${BLUE}Evaluating by region...${NC}"
python scripts/eval_by_region.py \
  --model $RESULTS_DIR/teacher_regional_v1/best_model.pt \
  --test_jsonl $DATA_DIR/dataset_splits.jsonl \
  --output $RESULTS_DIR/teacher_regional_v1/eval_by_region.json

echo -e "${BLUE}Computing teacher regional logits...${NC}"
python scripts/compute_teacher_logits.py \
  --model $RESULTS_DIR/teacher_regional_v1/best_model.pt \
  --train_jsonl $DATA_DIR/dataset_splits.jsonl \
  --output $DATA_DIR/teacher_regional_logits_train.npz

echo -e "${GREEN}✓ Teacher regional ready${NC}\n"

# ============ FASE 3: Combine soft labels ============
echo -e "${BLUE}[Fase 3] Combining teacher logits for distillation...${NC}"
python scripts/combine_teacher_logits.py \
  --teacher_global $DATA_DIR/teacher_global_logits_train.npz \
  --teacher_regional $DATA_DIR/teacher_regional_logits_train.npz \
  --weights 0.5 0.5 \
  --output $DATA_DIR/soft_labels_combined_train.npz

echo -e "${GREEN}✓ Soft labels ready${NC}\n"

# ============ FASE 4: Student Distillation ============
echo -e "${BLUE}[Fase 4] Training student (distillation phase)...${NC}"
python scripts/train_student_distill.py \
  --config config/student.yaml \
  --soft_labels $DATA_DIR/soft_labels_combined_train.npz \
  --output_dir $RESULTS_DIR/student_distill_v1 \
  --seed $SEED

echo -e "${BLUE}Training student (fine-tuning phase)...${NC}"
python scripts/train_student_finetune.py \
  --config config/student.yaml \
  --checkpoint $RESULTS_DIR/student_distill_v1/best_model.pt \
  --output_dir $RESULTS_DIR/student_finetune_v1 \
  --seed $SEED

echo -e "${GREEN}✓ Student trained${NC}\n"

# ============ FASE 5: Evaluation & Calibration ============
echo -e "${BLUE}[Fase 5] Full evaluation...${NC}"
python scripts/full_evaluation.py \
  --model $RESULTS_DIR/student_finetune_v1/best_model.pt \
  --test_jsonl $DATA_DIR/dataset_splits.jsonl \
  --output $RESULTS_DIR/student_finetune_v1/full_evaluation.json

echo -e "${BLUE}Calibrating model...${NC}"
python scripts/calibrate_model.py \
  --model $RESULTS_DIR/student_finetune_v1/best_model.pt \
  --val_jsonl $DATA_DIR/dataset_splits.jsonl \
  --output $RESULTS_DIR/student_finetune_v1/calibration.json

echo -e "${BLUE}Finding confidence threshold...${NC}"
python scripts/find_confidence_threshold.py \
  --model $RESULTS_DIR/student_finetune_v1/best_model.pt \
  --val_jsonl $DATA_DIR/dataset_splits.jsonl \
  --output $RESULTS_DIR/student_finetune_v1/threshold.json

echo -e "${GREEN}✓ Evaluation complete${NC}\n"

# ============ FASE 6: Quantization & Export ============
echo -e "${BLUE}[Fase 6] Quantizing model...${NC}"
python scripts/quantize_model.py \
  --model $RESULTS_DIR/student_finetune_v1/best_model.pt \
  --quantization_type fp16 \
  --output $RESULTS_DIR/student_finetune_v1/model_fp16.pt

echo -e "${BLUE}Exporting to SavedModel...${NC}"
python scripts/export_to_savedmodel.py \
  --model $RESULTS_DIR/student_finetune_v1/model_fp16.pt \
  --output_dir $RESULTS_DIR/student_finetune_v1

echo -e "${BLUE}Converting to TF.js...${NC}"
python scripts/export_to_tfjs.py \
  --model $RESULTS_DIR/student_finetune_v1/model_fp16.pt \
  --output_dir ./dist/models/student_v1.0 \
  --format tfjs \
  --quantization fp16

echo -e "${BLUE}Validating TF.js export...${NC}"
python scripts/validate_tfjs_export.py \
  --original_model $RESULTS_DIR/student_finetune_v1/model_fp16.pt \
  --tfjs_model ./dist/models/student_v1.0

echo -e "${GREEN}✓ Export complete${NC}\n"

# ============ FASE 7: Reproducibility ============
echo -e "${BLUE}[Fase 7] Generating reproducibility manifest...${NC}"
python scripts/generate_manifest.py \
  --phase_results $RESULTS_DIR \
  --output $RESULTS_DIR/TRAINING_MANIFEST_v1.0.yaml

echo -e "${BLUE}License compliance check...${NC}"
python scripts/license_checker.py \
  --manifest $RESULTS_DIR/TRAINING_MANIFEST_v1.0.yaml

echo -e "${BLUE}Creating dashboard...${NC}"
python scripts/dashboard.py \
  --evaluation_json $RESULTS_DIR/student_finetune_v1/full_evaluation.json \
  --output $RESULTS_DIR/eval_dashboard.png

echo -e "${GREEN}✓ Reproducibility complete${NC}\n"

# ============ Final Summary ============
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ PIPELINE COMPLETE${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results: $RESULTS_DIR"
echo "Model: ./dist/models/student_v1.0"
echo "Manifest: $RESULTS_DIR/TRAINING_MANIFEST_v1.0.yaml"
echo ""
echo "Next: Deploy to PWA (see EXPORT_TFJS_PWA.md)"
```

### Ejecutar

```bash
chmod +x scripts/reproduce_full_pipeline.sh
./scripts/reproduce_full_pipeline.sh

# Si algo falla, puedes reanudar desde una fase específica
# (cada script es independiente si tienes checkpoints)
```

---

## 5. CLI Arguments Pattern

Todos los scripts siguen este patrón:

```bash
python scripts/<script>.py \
  --config <yaml> \
  --input <path> \
  --output <path> \
  --seed <int> \
  --log_level <DEBUG|INFO|WARNING|ERROR> \
  --device <cuda|cpu> \
  --dry_run  # Simular sin guardar
```

### Ejemplo: script personalizado

```python
# scripts/train_teacher.py

import argparse
import logging
from pathlib import Path
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger.info(f"Config loaded: {args.config}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Device: {args.device}")

    if args.dry_run:
        logger.warning("DRY RUN: not saving anything")
        return

    # ... rest of training code

if __name__ == "__main__":
    main()
```

---

## 6. .gitignore

```
# Data (demasiado grande)
data/
checkpoints/
results/
dist/
*.npz
*.pt
*.h5
*.onnx

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
venv/
.venv

# Logs
logs/
*.log

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Secrets
.env
.env.local
secrets/

# Temporary
*.tmp
test_tfjs.js
tfjs_predictions.json
```

---

## 7. Testing

```python
# tests/test_distillation.py

import pytest
import torch
from models.student_architecture import StudentPlantRecognition
from models.distillation_loss import DistillationLoss

def test_student_forward():
    model = StudentPlantRecognition(num_classes=100)
    batch = torch.randn(4, 3, 224, 224)
    output = model(batch)
    assert output.shape == (4, 100)

def test_distillation_loss():
    loss_fn = DistillationLoss(temperature=3.0, alpha=0.7)
    student_logits = torch.randn(4, 100)
    teacher_logits = torch.randn(4, 100)
    hard_labels = torch.tensor([0, 1, 2, 3])

    loss = loss_fn(student_logits, teacher_logits, hard_labels)
    assert loss.item() >= 0

# Ejecutar
pytest tests/ -v
```

---

## 8. Resumen

**Para entrenar:**
```bash
./scripts/reproduce_full_pipeline.sh
```

**Para un script individual:**
```bash
python scripts/train_teacher.py --config config/teacher_global.yaml --output_dir ./results
```

**Para ver logs:**
```bash
tensorboard --logdir ./logs
```

**Para reproducir exactamente:**
```bash
cat ./results/TRAINING_MANIFEST_v1.0.yaml  # Ver parámetros exactos
# Luego: ./scripts/reproduce_full_pipeline.sh (idéntico)
```
