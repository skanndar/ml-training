# Plan de Entrenamiento: 7 Fases

> Estado dic 2025: Fase 0 completada (ver `../PHASE_0_COMPLETE.md`). Fase 1 ejecutada de punta a punta y actualmente en re-run con mejoras 384px + smart crop (ver `../TRAINING_ANALYSIS_EPOCH15.md` y `../SUMMARY_SESSION_20251220.md`).

## Fase 0: Auditoría Dataset + Baseline

**Objetivo:** Validar datos y establecer métricas base.

**Duración esperada:** 1-2 días

**Estado (17 dic 2025):** ✅ Completada. Se ejecutó `scripts/export_dataset.py` con filtrado de licencias, `scripts/split_dataset.py` y se documentó en `ml-training/PHASE_0_COMPLETE.md`.

- 171,381 imágenes únicas (de 340,749 exportadas inicialmente)
- Train/Val/Test: 136,312 / 17,372 / 17,697 registros
- 7,120 clases en train, 4,147 en val
- 92.4% de las imágenes de train son licencias permisivas (CC0/CC-BY/CC-BY-SA)
- Outputs: `data/dataset_{raw,train,val,test}.jsonl`, `data/dataset_splits.jsonl`, `data/dataset_eu_sw_train.jsonl` (subset base) + `data/dataset_eu_sw_*_stratified.jsonl` (ver Fase 2)

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

### 1.0 - Preparar split estratificado (nuevo 20 dic 2025)

`scripts/create_stratified_split.py` recorre `data/dataset_raw.jsonl`, filtra solo imágenes cacheadas y reparte cada clase de forma estratificada (1 imagen → solo train, 2 imágenes → 1/1, 3+ → 80/10/10). Este proceso genera `data/dataset_{train,val,test}_stratified.jsonl` y garantiza que todas las clases de validation aparezcan en train.

```bash
cd ~/ml-training
source venv/bin/activate
python scripts/create_stratified_split.py
```

**Resultados esperados:**

- Train: 93,761 imágenes | 5,914 clases
- Val: 12,639 imágenes | 4,410 clases
- Test: 11,726 imágenes
- Overlap train-val: 100% (0 clases exclusivas en val)

> Referencia: `SUMMARY_SESSION_20251220.md`

### 1.1 - Configuración

```yaml
# config/teacher_global.yaml

model:
  name: "vit_base_patch16_384"  # Cambiado desde 224px
  pretrained: true

training:
  learning_rate: 3.0e-5
  batch_size: 4  # Ajustado a GPU 4GB/384px
  epochs: 10
  warmup_epochs: 2
  optimizer: adamw
  weight_decay: 0.05
  gradient_clip: 1.0
  label_smoothing: 0.1

data:
  train_jsonl: "./data/dataset_train_stratified.jsonl"
  val_jsonl: "./data/dataset_val_stratified.jsonl"
  image_size: 384
  cache_size_gb: 220
  num_workers: 6
  smart_crop: true  # Saliency crop antes de augmentations

augmentation:
  train:
    resize: 448
    crop: 384
    horizontal_flip: 0.5
    vertical_flip: 0.2
    rotation: 20
    color_jitter:
      brightness: 0.3
      contrast: 0.3
      saturation: 0.3
      hue: 0.15
    blur: 0.1
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  val:
    resize: 448
    center_crop: 384
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]

regularization:
  dropout: 0.3
  mixup_alpha: 1.0
  cutmix_alpha: 1.2
  mixup_prob: 0.5

callbacks:
  early_stopping: true
  early_stopping_patience: 3
  early_stopping_metric: top1_acc
  save_best: true
  save_checkpoint_every: 1
  checkpoint_dir: "./checkpoints/teacher_global"
```

### 1.2 - Ejecutar entrenamiento

```bash
# Validar splits y lanzar 384px + smart crop
cd ~/ml-training
./START_TRAINING_384.sh

# El script:
# - Activa venv y asegura `dataset_*_stratified.jsonl`
# - Respalda checkpoints incompatibles (224px)
# - Lanza training con nohup → `training_384_smartcrop.log`

# Ejecución manual (opcional)
python scripts/train_teacher.py \
  --config config/teacher_global.yaml \
  --seed 42

# Monitoreo recomendado
tail -f training_384_smartcrop.log | grep -E 'Epoch|loss|top1'
tensorboard --logdir ./checkpoints/teacher_global/logs --port 6006
```

### 1.3 - Evaluación

```bash
python scripts/eval_teacher.py \
  --model ./results/teacher_global_v1/best_model.pt \
  --test_jsonl ./data/dataset_test_stratified.jsonl \
  --split_name "test" \
  --output ./results/teacher_global_v1/eval_test_stratified.json

# Output esperado:
# {
#   "top1_accuracy": 0.72,
#   "top5_accuracy": 0.88,
#   "loss": 0.98,
#   "per_region": { "EU_SW": 0.75, ... }
#   "stratified": true
# }
```

### 1.4 - Criterios de éxito

- [ ] Top-1 >= 70% (test set)
- [ ] Top-5 >= 85%
- [ ] Validation loss estable (sin overfitting severo)
- [ ] Per-class recall > 60% (incluso para clases con <10 imágenes)

> Nota (20 dic 2025): la corrida ViT-Base 224px completó 15 épocas con 98.6% Top-1 en train pero 0.03% en val (`TRAINING_ANALYSIS_EPOCH15.md`) por el split aleatorio. Con el split estratificado + 384px esperamos ver Top-1 val >5% en las primeras épocas y >15% al cierre de la corrida antes de continuar con el rollout global.

### 1.5 - Guardar logits (para distillation)

```bash
python scripts/compute_teacher_logits.py \
  --model ./results/teacher_global_v1/best_model.pt \
  --train_jsonl ./data/dataset_train_stratified.jsonl \
  --output ./data/teacher_global_logits_train.npz

# Guarda: {
#   "logits": (93761, 5914),  # predicciones sobre train estratificado
#   "indices": (93761,),      # plant IDs
#   "confidences": (93761,)
# }
```

### 1.6 - Estado actual (20 dic 2025)

- ✅ **Corrida ViT-Base 224px completada** (15 épocas). Resultado: overfitting severo por 186 clases exclusivas en validation y 27.7% de clases con solo 1 imagen (`TRAINING_ANALYSIS_EPOCH15.md`).
- ✅ **Dataset actualizado** con 741,533 imágenes (365k iNat + 376k legacy) y caché validado (`SUMMARY_SESSION_20251220.md`).
- ✅ **Stratified split** (`scripts/create_stratified_split.py`) garantiza overlap 100% train/val y tamaños 93,761 / 12,639 / 11,726.
- ✅ **Mejoras de entrenamiento**: smart crop basado en saliencia (`models/smart_crop.py`), resolución 384px, regularización + mixup/cutmix, y script `START_TRAINING_384.sh` para automatizar backups + launch (`IMPROVEMENTS_384_SMARTCROP.md`).
- 🔄 **Acción actual**: re-entrenar el teacher global con la nueva config y monitorizar `training_384_smartcrop.log`/TensorBoard. No pasar a Fase 2 hasta contar con métricas de validación estables.

---

## Fase 2: Entrenar Teacher Regional (SW Europa)

**Objetivo:** Replicar las mejoras de Fase 1 (split estratificado + smart crop + 384px + regularización agresiva) enfocadas a EU_SW para ganar +15pp en precisión local.

**Duración esperada:** 1-2 días (dataset < 25k imágenes → epochs ~35-40 min en GPU 8GB).

### 2.0 - Generar split estratificado EU_SW

Reutiliza `scripts/create_stratified_split.py` con `--region EU_SW` para garantizar que TODAS las clases que aparecen en validation también existen en train.

```bash
cd ~/ml-training
source venv/bin/activate

python scripts/create_stratified_split.py \
  --input ./data/dataset_raw.jsonl \
  --output-prefix ./data/dataset_eu_sw \
  --region EU_SW \
  --cache-dir ./data/image_cache  # o el path definido en config/paths.yaml

# Salida:
#   dataset_eu_sw_train_stratified.jsonl
#   dataset_eu_sw_val_stratified.jsonl
#   dataset_eu_sw_test_stratified.jsonl
#   dataset_eu_sw_stratified_stats.json (resumen de clases)
```

> `SUMMARY_SESSION_20251220.md` documenta tamaños esperados (train ≈ 18k, val ≈ 2.4k, test ≈ 2.3k, todas cacheadas con smart rate limiting activo).

### 2.1 - Configuración 384px + Smart Crop

```yaml
# config/teacher_regional.yaml (resumen)

model:
  name: "vit_base_patch16_384"
  pretrained: true
  init_from: "./checkpoints/teacher_global/best_model.pt"  # opcional pero recomendado

training:
  learning_rate: 2.0e-5
  batch_size: 4
  epochs: 12
  warmup_epochs: 2
  optimizer: adamw
  weight_decay: 0.05
  gradient_clip: 1.0
  label_smoothing: 0.1

data:
  train_jsonl: "./data/dataset_eu_sw_train_stratified.jsonl"
  val_jsonl: "./data/dataset_eu_sw_val_stratified.jsonl"
  image_size: 384
  cache_size_gb: 220
  smart_crop: true
  region_filter: "EU_SW"

augmentation:
  train:
    resize: 448
    crop: 384
    horizontal_flip: 0.5
    vertical_flip: 0.2
    rotation: 20
    color_jitter: { brightness: 0.3, contrast: 0.3, saturation: 0.3, hue: 0.15 }
    blur: 0.1
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  val:
    resize: 448
    center_crop: 384
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]

regularization:
  dropout: 0.3
  mixup_alpha: 1.0
  cutmix_alpha: 1.2
  mixup_prob: 0.5

callbacks:
  early_stopping_metric: top1_acc
  save_checkpoint_every: 1
  save_best: true
```

### 2.2 - Lanzar entrenamiento con helper script

```bash
cd ~/ml-training
./START_TRAINING_REGIONAL_384.sh

# El helper:
# - Obtiene cache_dir de config/paths.yaml
# - Ejecuta create_stratified_split.py --region EU_SW si faltan JSONL
# - Respalda checkpoints previos
# - Lanza train_teacher.py --config config/teacher_regional.yaml (nohup)
# - Log: training_regional_384.log / TensorBoard: checkpoints/teacher_regional/logs
```

Monitorea con:

```bash
tail -f training_regional_384.log | grep -E 'Epoch|loss|top'
tensorboard --logdir ./checkpoints/teacher_regional/logs --port 6007
```

### 2.3 - Evaluación EU_SW

```bash
python scripts/eval_teacher.py \
  --model ./results/teacher_regional_v1/best_model.pt \
  --test_jsonl ./data/dataset_eu_sw_test_stratified.jsonl \
  --split_name "test" \
  --output ./results/teacher_regional_v1/eval_test_eu_sw.json

python scripts/eval_by_region.py \
  --model ./results/teacher_regional_v1/best_model.pt \
  --test_jsonl ./data/dataset_test_stratified.jsonl \
  --output ./results/teacher_regional_v1/eval_by_region.json
```

### 2.4 - Criterios de éxito

- [ ] Top-1 EU_SW >= 80% (meta principal del teacher regional)
- [ ] Gap vs teacher global en EU_SW ≥ +10 pp
- [ ] Loss/Top-1 de val estable (sin colapso como en Fase 1 gracias al split estratificado)

### 2.5 - Guardar logits regionales

```bash
python scripts/compute_teacher_logits.py \
  --model ./results/teacher_regional_v1/best_model.pt \
  --train_jsonl ./data/dataset_eu_sw_train_stratified.jsonl \
  --output ./data/teacher_regional_logits_train.npz
```

> **Estado (22 dic, 21:15):** el entrenamiento EU_SW va por la época 4 con `Val Top-1 ≈ 89.9%`. `best_model.pt` y `last_checkpoint.pt` ya están generados, por lo que puedes parar cuando quieras y ejecutar inmediatamente los comandos de evaluación/logits anteriores.


## Fase 2b: Teacher C (Europa Norte/East)

**Objetivo:** cubrir las regiones EU_NORTH + EU_EAST para complementar al teacher SW.

### 2b.0 - Generar split EU_CORE

```bash
cd ~/ml-training
source venv/bin/activate

python scripts/create_stratified_split.py \
  --input ./data/dataset_raw.jsonl \
  --output-prefix ./data/dataset_eu_core \
  --region EU_NORTH,EU_EAST \
  --cache-dir ./data/image_cache

# Salida: dataset_eu_core_{train,val,test}_stratified.jsonl + dataset_eu_core_stratified_stats.json
```

### 2b.1 - Configuración

```yaml
# config/teacher_eu_core.yaml

model:
  name: "vit_base_patch16_384"
  pretrained: true
  init_from: "./checkpoints/teacher_global/best_model.pt"

data:
  train_jsonl: "./data/dataset_eu_core_train_stratified.jsonl"
  val_jsonl: "./data/dataset_eu_core_val_stratified.jsonl"
  image_size: 384
  smart_crop: true
  cache_size_gb: 220
```

### 2b.2 - Lanzar entrenamiento

```bash
cd ~/ml-training
./START_TRAINING_EU_CORE_384.sh

# Monitoreo
tail -f training_eu_core_384.log | grep -E 'Epoch|loss|top'
tensorboard --logdir ./checkpoints/teacher_eu_core/logs --port 6008
```

> **Estado (22 dic):** Teacher C (EU_CORE) en marcha con la misma configuración; si ves que las mejoras se estabilizan tras la época 4‑5, reduce `training.epochs` a 6‑8 para ahorrar tiempo.

### 2b.3 - Evaluación y logits

```bash
python scripts/eval_teacher.py \
  --model ./results/teacher_eu_core_v1/best_model.pt \
  --test_jsonl ./data/dataset_eu_core_test_stratified.jsonl \
  --output ./results/teacher_eu_core_v1/eval_test.json

python scripts/compute_teacher_logits.py \
  --model ./results/teacher_eu_core_v1/best_model.pt \
  --train_jsonl ./data/dataset_eu_core_train_stratified.jsonl \
  --output ./data/teacher_eu_core_logits_train.npz
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

- [x] Soft labels shape correcto: (≈58k, 8.6k clases, unión teachers)
- [x] Soft probs suman a 1.0 (tras combinación con pesos 0.4/0.4/0.2)
- [ ] Student architecture peso < 20MB (pendiente tras fine-tuning/export)
- [x] Checkpoints guardables (`checkpoints/student_distill/best_model.pt`)

> **Estado (23 dic 2025):** distillation completa (20 épocas) con `soft_labels_combined_train.npz` (58,280 muestras, 8,587 clases). Resultado: `Val Top-1 = 78.13%`, `Top-5 = 84.44%`, `Loss = 7.23` en la época 20. Curva saturó a partir de la época 15, por lo que los parámetros actuales (alpha 0.7 / beta 0.3, MobileNetV2 224px) son suficientes para pasar a fine-tuning.

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
  --checkpoint ./checkpoints/student_distill/best_model.pt \
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

- [x] Validation accuracy >= 70% (Val Top-1 = 78.13%)
- [x] Validation loss convergida (se estabiliza en ~7.2)
- [x] Training accuracy > 80% (Top-1 > 94%)
- [x] Checkpoints guardados cada epoch

> **Siguiente paso (23 dic 2025):** usar el checkpoint del student distillado (`checkpoints/student_distill/best_model.pt`, Val Top-1 78%) como punto de partida para el script `train_student_finetune.py` (10 épocas adicionales con CE pura y augmentaciones agresivas) para exprimir unos puntos extra antes de calibración/export.

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
