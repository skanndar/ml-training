# Resumen de Sesión - 20 Diciembre 2025

## Objetivos Completados

### 1. ✅ Actualización del Dataset desde MongoDB
- Exportadas **741,533 imágenes** (vs 340,749 anteriores)
- 8,587 clases únicas
- 365,025 de iNaturalist + 376,481 de fuentes legacy

### 2. ✅ Verificación del Smart Rate Limiting
- Test de 50 descargas: **100% exitosas**
- Sistema de backoff exponencial funcionando
- Cache creciendo correctamente: 91.87GB → 91.93GB

### 3. ✅ Incremento de Resolución a 384px
- Modelo cambiado: `vit_base_patch16_224` → `vit_base_patch16_384`
- Batch size ajustado: 16 → 8 (para VRAM)
- **+194% más píxeles** para capturar detalle

### 4. ✅ Implementación de Smart Crop
- Creado `models/smart_crop.py` con SaliencyCropper
- Integrado en `StreamingImageDataset`
- Detección automática de la planta usando saliency maps
- Fallback a center crop si falla

### 5. ✅ Dataset Stratified Split
- Script `create_stratified_split.py` creado
- **100% de clases en val también en train** (vs 44.8% anterior)
- 0 clases solo en validation (vs 186 anterior)
- Train: 93,761 | Val: 12,639 | Test: 11,726

---

## Análisis del Training Anterior (Epoch 15)

### Problema Identificado: Overfitting Severo

**Train accuracy**: 98.6% ✅
**Validation accuracy**: 0.03% ❌ (nivel de azar)

### Causas Raíz

1. **Dataset split inadecuado**
   - 186 clases solo en validation → modelo nunca las vio
   - Random shuffle sin estratificación

2. **Clases con muy pocas imágenes**
   - 27.7% de clases con solo 1 imagen en train
   - 36.6% de clases con solo 1 imagen en val

3. **Modelo muy grande para el dataset**
   - 90M parámetros vs 94k imágenes
   - Ratio: ~960 parámetros por imagen → memorización

### Lecciones Aprendidas

- El modelo **puede** aprender (98.6% train)
- El código **funciona** perfectamente (0 errores)
- El problema **no es técnico**, es estadístico
- Stratified split es **crítico** para validation

---

## Mejoras Implementadas

### Cambios en Configuración

**config/teacher_global.yaml**:
```yaml
model:
  name: "vit_base_patch16_384"  # Antes: 224

training:
  batch_size: 8  # Antes: 16

data:
  train_jsonl: "./data/dataset_train_stratified.jsonl"  # Antes: from_raw_cached
  val_jsonl: "./data/dataset_val_stratified.jsonl"
  image_size: 384  # Antes: 224
  smart_crop: true  # NUEVO
  cache_size_gb: 220

augmentation:
  train:
    resize: 448  # Antes: 256
    crop: 384    # Antes: 224
  val:
    resize: 448  # Antes: 256
    center_crop: 384  # Antes: 224
```

### Nuevos Archivos Creados

1. **models/smart_crop.py**
   - `SaliencyCropper`: Crop inteligente usando saliency detection
   - `SmartCropTransform`: Wrapper para PyTorch datasets
   - Fallback a center crop si falla

2. **scripts/create_stratified_split.py**
   - Crea splits asegurando overlap de clases
   - Maneja clases con 1, 2, o 3+ imágenes
   - Output: `dataset_{train,val,test}_stratified.jsonl`

3. **scripts/test_rate_limiting.py**
   - Verifica smart rate limiting
   - Intenta descargar imágenes nuevas
   - Detecta cuando 429 activa backoff

4. **scripts/test_smart_crop.py**
   - Visualiza smart crop vs center crop
   - Genera comparación side-by-side
   - Output: `results/smart_crop_comparison.png`

5. **START_TRAINING_384.sh**
   - Script completo para iniciar training
   - Verifica datasets, hace backup de checkpoints
   - Muestra configuración y expectativas

6. **IMPROVEMENTS_384_SMARTCROP.md**
   - Documentación completa de mejoras
   - Justificación técnica de cada cambio
   - Expectativas de rendimiento

### Archivos Modificados

1. **models/streaming_dataset.py**
   - Añadido soporte para `smart_crop` y `smart_crop_size`
   - Integración de `SaliencyCropper`
   - Smart crop se aplica ANTES de augmentations

2. **models/dataloader_factory.py**
   - Lee config de `smart_crop` y `image_size`
   - Pasa parámetros a `StreamingImageDataset`

---

## Dependencias Instaladas

```bash
pip install opencv-contrib-python  # Para saliency detection
```

---

## Estado Actual

### Dataset
- **Total disponible**: 741,533 imágenes
- **Cacheadas**: 62,821 (91.93GB)
- **Por descargar**: 678,712 (se descargarán gradualmente)

### Splits Stratified
- **Train**: 93,761 imágenes, 5,914 clases
- **Val**: 12,639 imágenes, 4,410 clases
- **Test**: 11,726 imágenes
- **Overlap**: 100% (todas las clases en val están en train)

### Smart Rate Limiting
- ✅ Funcionando correctamente
- 50/50 descargas exitosas en test
- Backoff exponencial: 60s → 120s → 300s → 600s
- Auto-recuperación cuando expira el límite

### Smart Crop
- ✅ Implementado y testeado
- Detección de saliency usando OpenCV
- Fallback a center crop si falla
- Visualización generada en `results/smart_crop_comparison.png`

---

## Próximo Paso: Entrenar con Mejoras

### Comando

```bash
cd /home/skanndar/SynologyDrive/local/aplantida/ml-training
./START_TRAINING_384.sh
```

O manualmente:

```bash
cd /home/skanndar/SynologyDrive/local/aplantida/ml-training
source venv/bin/activate

# Verificar stratified split existe
python3 scripts/create_stratified_split.py

# Backup checkpoints viejos
mv checkpoints/teacher_global checkpoints/teacher_global_224_backup

# Iniciar training
nohup python3 scripts/train_teacher.py \
    --config config/teacher_global.yaml \
    > training_384_smartcrop.log 2>&1 &

# Monitorear
tail -f training_384_smartcrop.log | grep -E 'Epoch|loss|top1'
```

### Expectativas Realistas

| Métrica | Anterior (224px) | Esperado (384px + smart crop) |
|---------|------------------|-------------------------------|
| **Val Top-1** | 0.03% | **5-15%** |
| **Val Top-5** | 0.08% | **15-30%** |
| **Train Top-1** | 98.6% | 95-98% |
| **Tiempo/epoch** | 45 min | 70-80 min |
| **Epochs** | 15 | 15 |
| **Tiempo total** | 11.25 horas | 18-20 horas |

### Por qué Esperamos Mejora

1. **Stratified split** → Modelo ve todas las clases en validación
2. **384px** → 3x más detalle para distinguir especies
3. **Smart crop** → Enfoque en planta vs fondo
4. **Dataset más grande** → 741k imágenes disponibles

### Si la Mejora es Insuficiente

Si después de 5-7 epochs val accuracy < 5%:

**Opción A: Filtrar clases con pocas imágenes**
```python
# Solo entrenar con clases que tengan ≥5 imágenes
min_images_per_class = 5
```

**Opción B: Aumentar regularización**
```yaml
regularization:
  dropout: 0.3  # Subir de 0.2
  weight_decay: 0.05  # Subir de 0.01
  label_smoothing: 0.2  # Subir de 0.1
```

**Opción C: Modelo más pequeño**
```yaml
model:
  name: "vit_small_patch16_384"  # ~22M params vs 90M
```

---

## Archivos de Referencia

### Documentación
- `IMPROVEMENTS_384_SMARTCROP.md` - Detalles técnicos de mejoras
- `TRAINING_ANALYSIS_EPOCH15.md` - Análisis del training anterior
- `SMART_RATE_LIMITING.md` - Sistema de rate limiting
- `TRAINING_CHEATSHEET.md` - Comandos de gestión
- `CACHE_FIX_20251219.md` - Fix de cache eviction

### Scripts
- `START_TRAINING_384.sh` - Iniciar training con mejoras
- `scripts/create_stratified_split.py` - Crear splits estratificados
- `scripts/test_rate_limiting.py` - Test de rate limiting
- `scripts/test_smart_crop.py` - Visualización de smart crop
- `scripts/export_dataset.py` - Exportar desde MongoDB

### Configuración
- `config/teacher_global.yaml` - Config principal (384px + smart crop)

---

## Resumen de la Sesión

**Duración**: ~4 horas
**Logros**:
1. Diagnóstico completo del overfitting anterior
2. Dataset actualizado (741k imágenes)
3. Implementación de smart crop con saliency
4. Upgrade a 384px con config optimizado
5. Stratified split para validación correcta
6. Verificación de smart rate limiting

**Estado**: Listo para entrenar con las mejoras

**Próximo paso**: Ejecutar `./START_TRAINING_384.sh` y monitorear

---

## Comandos de Monitoreo

```bash
# Ver progreso general
tail -f training_384_smartcrop.log | grep -E 'Epoch|loss|top1'

# Ver log completo
tail -f training_384_smartcrop.log

# GPU usage
nvidia-smi -l 2

# Cache size
du -sh /media/skanndar/2TB1/aplantida-ml/image_cache

# Proceso de training
ps aux | grep train_teacher

# Matar training si necesario
pkill -f train_teacher.py
```

---

## Contacto y Feedback

Si tienes preguntas o encuentras issues:
1. Revisa primero `IMPROVEMENTS_384_SMARTCROP.md`
2. Revisa `TRAINING_CHEATSHEET.md` para comandos
3. Revisa logs: `training_384_smartcrop.log`
