# Plan de Entrenamiento: 7 Fases

## Fase 0: Auditoría Dataset + Baseline

**Objetivo:** Validar datos y establecer métricas base.

**Duración esperada:** 1-2 días

### 0.1 - Ejecutar

```bash
cd ~/ml-training  # Repo principal

# Auditar Mongo + validar URLs
python scripts/audit_dataset.py
python scripts/validate_image_urls.py

# Exportar a JSONL + hacer splits
python scripts/export_dataset.py --output_dir ./data
python scripts/split_dataset.py --jsonl_path ./data/dataset_raw.jsonl --output_dir ./data

# Baseline con modelo pretrained (sin fine-tuning)
python scripts/baseline.py \
  --model_name mobilenetv2_1.0_224 \
  --test_jsonl ./data/dataset_splits.jsonl \
  --output ./results/baseline_v0
```

### 0.2 - Criterios de éxito

- [ ] Todas las URLs descargables (tasa fallo < 5%)
- [ ] Split train/val/test sin leakage
- [ ] ~800k-900k imágenes válidas
- [ ] Balance por clase documentado (histograma)
- [ ] Baseline Top-1: ~30-40% (esperado para model genérico)

### 0.3 - Riesgos & mitigaciones

| Riesgo | Solución |
|--------|----------|
| URLs broken | Reentrentar solo con URLs válidas |
| Desbalanceo extremo | Usar weighted sampling en loader |
| Imágenes muy pequeñas (<100px) | Filtrar en scripts/export_dataset.py |

---

## Fase 1: Entrenar Teacher Global

**Objetivo:** Fine-tune modelo global en dataset completo.

**Duración esperada:** 2-3 días (depende GPU)

### 1.1 - Configuración

```yaml
# config/teacher_global.yaml

model:
  name: "vit_base_patch16_224"
  pretrained: true
  num_classes: 9000  # Ajustar según Mongo

training:
  learning_rate: 1e-5
  batch_size: 64
  epochs: 15
  warmup_epochs: 2
  weight_decay: 0.01

data:
  train_jsonl: "./data/dataset_splits.jsonl"
  split_name: "train"
  augmentation: "aggressive"  # Ver DATOS_PIPELINE.md
  image_size: 224

regularization:
  dropout: 0.2
  mixup_alpha: 0.8
  cutmix_alpha: 0.8

callbacks:
  early_stopping: true
  early_stopping_patience: 3
  save_best: true
  checkpoint_dir: "./checkpoints/teacher_global"
```

### 1.2 - Ejecutar entrenamiento

```bash
# Entrenar
python scripts/train_teacher.py \
  --config config/teacher_global.yaml \
  --output_dir ./results/teacher_global_v1 \
  --seed 42

# Este comando:
# - Carga dataset con splits
# - Fine-tunea ViT por 15 épocas
# - Guarda checkpoints cada epoch
# - Log: loss, top1 acc, top5 acc en train/val
# - Output: teacher_global_v1.pt + teacher_global_v1_metrics.json
```

### 1.3 - Evaluación

```bash
python scripts/eval_teacher.py \
  --model ./results/teacher_global_v1/best_model.pt \
  --test_jsonl ./data/dataset_splits.jsonl \
  --split_name "test" \
  --output ./results/teacher_global_v1/eval_test.json

# Output esperado:
# {
#   "top1_accuracy": 0.72,
#   "top5_accuracy": 0.88,
#   "loss": 0.98,
#   "per_region": {
#     "EU_SW": 0.75,
#     "EU": 0.70,
#     ...
#   }
# }
```

### 1.4 - Criterios de éxito

- [ ] Top-1 >= 70% (test set)
- [ ] Top-5 >= 85%
- [ ] Validation loss estable (sin overfitting severo)
- [ ] Per-class recall > 60% (incluso para clases con <10 imágenes)

### 1.5 - Guardar logits (para distillation)

```bash
python scripts/compute_teacher_logits.py \
  --model ./results/teacher_global_v1/best_model.pt \
  --train_jsonl ./data/dataset_splits.jsonl \
  --output ./data/teacher_global_logits_train.npz

# Guarda: {
#   "logits": (850000, 9000),  # train set predictions
#   "indices": (850000,),       # plant IDs
#   "confidences": (850000,)
# }
```

---

## Fase 2: Entrenar Teacher Regional (SW Europa)

**Objetivo:** Fine-tune en subset de SW Europa para mejorar precision local.

**Duración esperada:** 1-2 días

### 2.1 - Preparar subset regional

```python
# scripts/prepare_regional_dataset.py

import pandas as pd

df = pd.read_json("./data/dataset_splits.jsonl", lines=True)

# Filtrar región SW Europa
regional_df = df[(df["region"] == "EU_SW") & (df["split"] == "train")]

print(f"Regional samples: {len(regional_df)}")
print(f"Regional species: {len(regional_df['latin_name'].unique())}")

# Guardar
regional_df.to_json("./data/dataset_eu_sw_train.jsonl", orient="records", lines=True)

# Stats
print(regional_df["latin_name"].value_counts().head(20))
```

### 2.2 - Entrenar teacher regional

```yaml
# config/teacher_regional.yaml

model:
  name: "vit_base_patch16_224"
  pretrained: true  # Empezar desde ImageNet
  num_classes: 9000  # Mismas clases

training:
  learning_rate: 5e-6  # Más bajo (dataset más pequeño)
  batch_size: 32
  epochs: 20
  warmup_epochs: 2

data:
  train_jsonl: "./data/dataset_eu_sw_train.jsonl"
  augmentation: "moderate"
  image_size: 224

regularization:
  dropout: 0.3
  weight_decay: 0.02
  early_stopping_patience: 5
```

```bash
python scripts/train_teacher.py \
  --config config/teacher_regional.yaml \
  --output_dir ./results/teacher_regional_eu_sw_v1 \
  --seed 42
```

### 2.3 - Evaluar por región

```bash
python scripts/eval_by_region.py \
  --model ./results/teacher_regional_eu_sw_v1/best_model.pt \
  --test_jsonl ./data/dataset_splits.jsonl \
  --split_name "test" \
  --output ./results/teacher_regional_v1/eval_by_region.json

# Output esperado:
# {
#   "EU_SW": {
#     "top1": 0.82,  # Mejor que global!
#     "top5": 0.92
#   },
#   "EU": {
#     "top1": 0.68,  # Peor en otras regiones
#     "top5": 0.85
#   }
# }
```

### 2.4 - Criterios de éxito

- [ ] Top-1 EU_SW >= 80% (regional boost)
- [ ] Top-1 test set global >= 65% (puede bajar respecto teacher global)
- [ ] Validación loss convergida

### 2.5 - Guardar logits regionales

```bash
python scripts/compute_teacher_logits.py \
  --model ./results/teacher_regional_eu_sw_v1/best_model.pt \
  --train_jsonl ./data/dataset_splits.jsonl \
  --output ./data/teacher_regional_logits_train.npz
```

---

## Fase 3: Diseñar Student + Preparar Distillation

**Objetivo:** Definir arquitectura student y generar soft labels.

**Duración esperada:** 1 día

### 3.1 - Definir arquitectura Student

```yaml
# config/student.yaml

model:
  name: "mobilenetv2"
  alpha: 1.0  # Width multiplier
  num_classes: 9000
  dropout_rate: 0.2

distillation:
  temperature: 3.0
  loss_weights:
    kl_divergence: 0.7  # Peso logits teachers
    cross_entropy: 0.3   # Peso labels reales

teachers:
  - path: "./results/teacher_global_v1/best_model.pt"
    weight: 0.5
  - path: "./results/teacher_regional_eu_sw_v1/best_model.pt"
    weight: 0.5
    region_filter: "EU_SW"  # Solo para muestras EU_SW

training:
  learning_rate: 1e-4
  batch_size: 128
  epochs: 20
  warmup_epochs: 1
  optimizer: "adam"
```

### 3.2 - Generar soft labels (on-the-fly vs precomputed)

#### Opción A: Precomputed (recomendado)

```bash
# Agregar logits de múltiples teachers
python scripts/combine_teacher_logits.py \
  --teacher_global ./data/teacher_global_logits_train.npz \
  --teacher_regional ./data/teacher_regional_logits_train.npz \
  --weights 0.5 0.5 \
  --output ./data/soft_labels_combined_train.npz

# Output: soft_labels_combined_train.npz
# ├─ soft_probs: (850k, 9000)  # Promedio ponderado logits
# └─ indices: (850k,)           # Para matching con dataset
```

#### Opción B: On-the-fly (si memoria es limitada)

```python
# En train_student.py:
def get_soft_labels_batch(batch_indices):
    # Cargar logits solo para batch actual
    logits_global = load_teacher_output(
        "./data/teacher_global_logits_train.npz",
        indices=batch_indices
    )
    logits_regional = load_teacher_output(
        "./data/teacher_regional_logits_train.npz",
        indices=batch_indices
    )
    combined = 0.5 * logits_global + 0.5 * logits_regional
    return tf.nn.softmax(combined / T)  # T = temperatura
```

### 3.3 - Criterios de éxito

- [ ] Soft labels shape correcto: (850k, 9000)
- [ ] Soft probs suman a 1.0
- [ ] Student architecture peso < 20MB
- [ ] Checkpoints guardables

---

## Fase 4: Entrenar Student (Distillation + Fine-tuning)

**Objetivo:** Entrenar student para aproximar ensemble de teachers.

**Duración esperada:** 3-5 días

### 4.1 - Distillation pura (Epochs 0-10)

```bash
python scripts/train_student_distill.py \
  --config config/student.yaml \
  --mode distillation_only \
  --epochs_phase1 10 \
  --soft_labels ./data/soft_labels_combined_train.npz \
  --output_dir ./results/student_distill_v1 \
  --seed 42

# Loss = KL(teacher_soft_probs || student_probs) + α * CE(hard_labels, student_logits)
# α = 0.3 (ver config/student.yaml)
```

### 4.2 - Fine-tuning con labels reales (Epochs 10-20)

```bash
python scripts/train_student_finetune.py \
  --checkpoint ./results/student_distill_v1/epoch_10/best_model.pt \
  --config config/student.yaml \
  --mode finetuning \
  --epochs_phase2 10 \
  --learning_rate 5e-5 \
  --augmentation aggressive \
  --output_dir ./results/student_finetune_v1 \
  --seed 42

# Loss = CE(hard_labels, student_logits) solamente
# LR más bajo, augmentation más fuerte
```

### 4.3 - Criterios de éxito

- [ ] Validation accuracy >= 70% (cercano a teachers)
- [ ] Validation loss convergida (no sube)
- [ ] Training accuracy > 80% (el modelo aprende)
- [ ] Checkpoints guardados cada epoch

### 4.4 - Monitorizar entrenamiento

```bash
# En tiempo real
tensorboard --logdir ./results/student_finetune_v1/logs

# Al finalizar
python scripts/plot_training_metrics.py \
  --student_logs ./results/student_finetune_v1/logs \
  --output ./results/student_finetune_v1/metrics_plot.png
```

---

## Fase 5: Calibración + Ajuste umbral

**Objetivo:** Calibrar confianzas y definir umbral "no conclusive".

**Duración esperada:** 1 día

### 5.1 - Evaluar calibración

```bash
python scripts/calibrate_model.py \
  --model ./results/student_finetune_v1/best_model.pt \
  --val_jsonl ./data/dataset_splits.jsonl \
  --output ./results/student_finetune_v1/calibration.json

# Output:
# {
#   "ece": 0.12,        # Expected Calibration Error
#   "mce": 0.25,        # Maximum Calibration Error
#   "predicted_probs": [...],
#   "true_labels": [...]
# }
```

### 5.2 - Si ECE > 0.1, aplicar temperature scaling

```bash
python scripts/temperature_scaling.py \
  --model ./results/student_finetune_v1/best_model.pt \
  --val_jsonl ./data/dataset_splits.jsonl \
  --output_temperature ./results/student_finetune_v1/temperature_scaled.pt

# Encuentra temperatura óptima para minimizar ECE
# Output: model con calibration mejorada
```

### 5.3 - Definir umbral "no conclusive"

```python
# scripts/find_confidence_threshold.py

# Análisis: ¿qué confianza mínima asegura >95% precisión?

import numpy as np

predictions = load_predictions(...)  # Top-1 confidence
true_labels = load_true_labels(...)

accuracy_by_threshold = []
for threshold in np.arange(0.4, 0.95, 0.05):
    mask = predictions >= threshold
    if mask.sum() > 0:
        acc = (predictions[mask] == true_labels[mask]).mean()
        coverage = mask.sum() / len(mask)
        accuracy_by_threshold.append({
            "threshold": threshold,
            "accuracy": acc,
            "coverage": coverage
        })

# Elegir threshold donde accuracy >= 0.95
# Típicamente: threshold = 0.70-0.80
```

### 5.4 - Criterios de éxito

- [ ] ECE < 0.10
- [ ] Coverage (samples con conf > threshold) >= 80%
- [ ] Precisión a umbral elegido >= 95%

---

## Fase 6: Export a TF.js + Optimizaciones

**Objetivo:** Convertir student a TF.js y optimizar tamaño/latencia.

**Duración esperada:** 1-2 días

### 6.1 - Cuantización

```bash
# Opción A: FP16 (trade-off recomendado)
python scripts/quantize_model.py \
  --model ./results/student_finetune_v1/best_model.pt \
  --quantization_type fp16 \
  --output ./results/student_finetune_v1/model_fp16.pt

# Opción B: INT8 (más agresivo)
python scripts/quantize_model.py \
  --model ./results/student_finetune_v1/best_model.pt \
  --quantization_type int8 \
  --calibration_data ./data/dataset_splits.jsonl \
  --output ./results/student_finetune_v1/model_int8.pt
```

### 6.2 - Export a TF.js

```bash
# Convertir PyTorch → ONNX → TF.js
python scripts/export_to_tfjs.py \
  --model ./results/student_finetune_v1/model_fp16.pt \
  --output_dir ./dist/models/student_v1.0 \
  --format tfjs \
  --quantization fp16

# Output:
# dist/models/student_v1.0/
# ├─ model.json
# ├─ group1-shard1of3.bin
# ├─ group1-shard2of3.bin
# └─ group1-shard3of3.bin
# (total ~70-100MB sin cuantizar, ~35-50MB con FP16)
```

### 6.3 - Validar equivalencia Python ↔ Browser

```bash
python scripts/validate_tfjs_export.py \
  --original_model ./results/student_finetune_v1/model_fp16.pt \
  --tfjs_model ./dist/models/student_v1.0 \
  --test_images ./data/test_samples.npz

# Compara predicciones:
# ├─ Python predictions: shape (100, 9000)
# ├─ TF.js predictions: shape (100, 9000)
# └─ Correlation: 0.9998 (excelente)
```

### 6.4 - Criterios de éxito

- [ ] Model size <= 150MB (cuantizado)
- [ ] Equivalencia Python ↔ TF.js > 99.5% correlation
- [ ] Inferencia < 2s en GPU, < 5s en CPU (browser)
- [ ] Archivos .bin descargables

---

## Fase 7: Reproducibilidad + Documentación

**Objetivo:** Documentar todo para futuras iteraciones.

**Duración esperada:** 1 día

### 7.1 - Crear reproducibility manifest

```yaml
# results/TRAINING_MANIFEST_v1.0.yaml

timestamp: "2024-12-17T15:30:00Z"
git_commit: "abc1234"

data:
  mongo_snapshot: "./data/dataset_splits.jsonl"
  total_samples: 850000
  total_species: 9000
  train_samples: 680000
  val_samples: 85000
  test_samples: 85000

phases:
  phase_0:
    status: "completed"
    baseline_top1: 0.35

  phase_1:
    status: "completed"
    teacher: "vit_base_patch16_224"
    top1: 0.72
    top5: 0.88
    checkpoint: "teacher_global_v1/best_model.pt"

  phase_2:
    status: "completed"
    teacher: "vit_base_regional_eu_sw"
    top1_eu_sw: 0.82
    checkpoint: "teacher_regional_v1/best_model.pt"

  phase_3:
    status: "completed"
    student_architecture: "mobilenetv2_1.0"

  phase_4:
    status: "completed"
    distillation_epochs: 10
    finetuning_epochs: 10
    final_top1: 0.71
    final_top5: 0.87
    checkpoint: "student_finetune_v1/best_model.pt"

  phase_5:
    status: "completed"
    ece: 0.09
    confidence_threshold: 0.75

  phase_6:
    status: "completed"
    model_size_fp16: 45  # MB
    tfjs_export: "dist/models/student_v1.0"

environment:
  python_version: "3.11"
  pytorch_version: "2.1.0"
  tensorflow_version: "2.14.0"
  gpu: "RTX 3090"
  seed: 42

hyperparameters:
  teacher_global_lr: 1e-5
  teacher_regional_lr: 5e-6
  student_distill_lr: 1e-4
  student_finetune_lr: 5e-5
  distillation_temperature: 3.0
  kl_weight: 0.7
  ce_weight: 0.3
```

### 7.2 - Crear script reproducible

```bash
#!/bin/bash
# scripts/reproduce_full_pipeline.sh

set -e

SEED=42
DATA_DIR="./data"
RESULTS_DIR="./results"

echo "Starting full training pipeline..."

# Fase 0
echo "Phase 0: Audit"
python scripts/audit_dataset.py
python scripts/split_dataset.py --jsonl_path ${DATA_DIR}/dataset_raw.jsonl

# Fase 1
echo "Phase 1: Train teacher global"
python scripts/train_teacher.py --config config/teacher_global.yaml --seed ${SEED}

# Fase 2
echo "Phase 2: Train teacher regional"
python scripts/train_teacher.py --config config/teacher_regional.yaml --seed ${SEED}

# Fase 3
echo "Phase 3: Prepare distillation"
python scripts/combine_teacher_logits.py

# Fase 4
echo "Phase 4: Train student"
python scripts/train_student_distill.py --seed ${SEED}
python scripts/train_student_finetune.py --seed ${SEED}

# Fase 5
echo "Phase 5: Calibrate"
python scripts/calibrate_model.py
python scripts/find_confidence_threshold.py

# Fase 6
echo "Phase 6: Export to TF.js"
python scripts/quantize_model.py --quantization_type fp16
python scripts/export_to_tfjs.py

# Fase 7
echo "Phase 7: Generate manifest"
python scripts/generate_manifest.py

echo "Pipeline complete!"
cat ${RESULTS_DIR}/TRAINING_MANIFEST_v1.0.yaml
```

### 7.3 - Criterios de éxito

- [ ] Manifest completo con todos los parámetros
- [ ] Script reproducible genera exactamente mismo model
- [ ] Seed fijo produce mismo accuracy
- [ ] Documentación de todos los hiperparámetros

---

## Resumen Checklist

| Fase | Objetivo | Éxito | ✓ |
|------|----------|-------|---|
| 0 | Auditoría dataset | Baseline ~35% | |
| 1 | Teacher global | Top-1 >= 70% | |
| 2 | Teacher regional | EU_SW +15% | |
| 3 | Distillation prep | Soft labels listos | |
| 4 | Student training | Top-1 >= 70% | |
| 5 | Calibración | ECE < 0.10 | |
| 6 | Export TF.js | Model <= 150MB | |
| 7 | Reproducibilidad | Script automatizado | |

---

## Troubleshooting rápido

**Loss no baja en Fase 4:**
- Reducir LR 10x
- Aumentar warmup
- Verificar soft labels (§ 3.2)

**GPU memory overflow:**
- Reducir batch_size a 32 (con gradient accumulation)
- Usar mixed precision (fp16)
- Activar gradient checkpointing en modelo

**Accuracy cae en val:**
- Posible overfitting: aumentar dropout/weight decay
- Datos train/val contaminados: revisar split (§ 2.1)
- Teacher logits incorrectos: regenerar (§ 1.5)
