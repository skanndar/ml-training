# Training Management Cheatsheet

## 🚀 Inicio Rápido - Training con 384px + Smart Crop

### Iniciar training con script automático (RECOMENDADO)
```bash
cd /home/skanndar/SynologyDrive/local/aplantida/ml-training
./START_TRAINING_384.sh
```

### Iniciar training manualmente
```bash
cd /home/skanndar/SynologyDrive/local/aplantida/ml-training
source venv/bin/activate

# Verificar que existen los datasets stratified
ls -lh data/dataset_*_stratified.jsonl

# Si no existen, crearlos
python3 scripts/create_stratified_split.py

# Iniciar training
nohup python3 scripts/train_teacher.py \
    --config config/teacher_global.yaml \
    > training_384_smartcrop.log 2>&1 &

# Ver progreso
tail -f training_384_smartcrop.log | grep -E 'Epoch|loss|top1'
```

---

## 📊 Monitorear Training

### Ver progreso en tiempo real
```bash
cd /home/skanndar/SynologyDrive/local/aplantida/ml-training

# Ver progreso filtrado (epochs, loss, accuracy)
tail -f training_384_smartcrop.log | grep -E 'Epoch|loss|top1'

# Ver log completo
tail -f training_384_smartcrop.log

# Ver últimas 100 líneas
tail -100 training_384_smartcrop.log

# Buscar errores
grep -i "error\|warning\|failed" training_384_smartcrop.log | tail -20
```

### Verificar estado del proceso
```bash
# Ver proceso de training
ps aux | grep train_teacher.py

# Ver uso de GPU
nvidia-smi

# Ver GPU continuamente (cada 2 segundos)
nvidia-smi -l 2

# Ver uso de GPU en formato compacto
watch -n 2 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader'
```

### Revisar caché
```bash
# Tamaño actual de la caché
du -sh /media/skanndar/2TB1/aplantida-ml/image_cache

# Número de imágenes en caché
find /media/skanndar/2TB1/aplantida-ml/image_cache -type f -name "*.jpg" | wc -l

# Espacio libre en disco
df -h /media/skanndar/2TB1/

# Ver las últimas imágenes descargadas
ls -lt /media/skanndar/2TB1/aplantida-ml/image_cache/*.jpg | head -20
```

### Ver estadísticas del dataset
```bash
# Contar imágenes por dataset
wc -l data/dataset_*_stratified.jsonl

# Ver primera línea de cada dataset
head -1 data/dataset_train_stratified.jsonl | python3 -m json.tool

# Verificar que smart_crop está activado
grep smart_crop config/teacher_global.yaml
```

---

## 🎮 Controlar Training

### Parar el training
```bash
# Opción 1: Parada elegante (espera a que termine el batch actual)
pkill -SIGTERM -f "python3 scripts/train_teacher.py"

# Opción 2: Parada inmediata (FORZAR)
pkill -9 -f "python3 scripts/train_teacher.py"

# Verificar que se detuvo
ps aux | grep train_teacher.py | grep -v grep
```

### Reanudar training desde ÚLTIMO CHECKPOINT
```bash
cd /home/skanndar/SynologyDrive/local/aplantida/ml-training
source venv/bin/activate

# Reanudar desde último checkpoint guardado
nohup python3 scripts/train_teacher.py \
    --config config/teacher_global.yaml \
    --resume checkpoints/teacher_global/last_checkpoint.pt \
    > training_384_smartcrop.log 2>&1 &

# Ver progreso
tail -f training_384_smartcrop.log | grep -E 'Epoch|loss|top1'
```

### Iniciar training DESDE CERO (eliminar checkpoints)
```bash
cd /home/skanndar/SynologyDrive/local/aplantida/ml-training
source venv/bin/activate

# Backup de checkpoints anteriores (por si acaso)
mv checkpoints/teacher_global checkpoints/teacher_global_backup_$(date +%Y%m%d_%H%M%S)

# Iniciar desde cero
nohup python3 scripts/train_teacher.py \
    --config config/teacher_global.yaml \
    > training_384_smartcrop.log 2>&1 &

# Ver progreso
tail -f training_384_smartcrop.log
```

---

## 🧪 Testing y Validación

### Test de Smart Crop (visualización)
```bash
cd /home/skanndar/SynologyDrive/local/aplantida/ml-training
source venv/bin/activate

# Generar comparación visual (6 ejemplos)
python3 scripts/test_smart_crop.py --samples 6

# Ver resultado
xdg-open results/smart_crop_comparison.png
# O abrir manualmente: results/smart_crop_comparison.png
```

### Test de Rate Limiting
```bash
cd /home/skanndar/SynologyDrive/local/aplantida/ml-training
source venv/bin/activate

# Test de descargas (50 imágenes nuevas)
python3 scripts/test_rate_limiting.py

# Ver si hay rate limiting activo
grep -i "rate limit" training_384_smartcrop.log | tail -10
```

### Debug de Saliency Detection
```bash
cd /home/skanndar/SynologyDrive/local/aplantida/ml-training
source venv/bin/activate

# Generar debug visual de saliency
python3 scripts/debug_saliency.py

# Ver resultado
xdg-open results/saliency_debug.png
```

---

## 📁 Gestión de Datasets

### Crear Stratified Split (si no existe)
```bash
cd /home/skanndar/SynologyDrive/local/aplantida/ml-training
source venv/bin/activate

python3 scripts/create_stratified_split.py

# Verificar resultados
wc -l data/dataset_*_stratified.jsonl
```

### Exportar dataset desde MongoDB
```bash
cd /home/skanndar/SynologyDrive/local/aplantida/ml-training
source venv/bin/activate

# Exportar todo el dataset
python3 scripts/export_dataset.py --output-dir data --min-images 1

# Ver estadísticas
cat data/export_stats.json | python3 -m json.tool
```

### Verificar overlap de clases (train vs val)
```bash
python3 << 'EOF'
import json
from collections import Counter

with open("data/dataset_train_stratified.jsonl") as f:
    train = [json.loads(line) for line in f]
with open("data/dataset_val_stratified.jsonl") as f:
    val = [json.loads(line) for line in f]

train_classes = set(Counter([r['class_idx'] for r in train]).keys())
val_classes = set(Counter([r['class_idx'] for r in val]).keys())

print(f"Train classes: {len(train_classes):,}")
print(f"Val classes: {len(val_classes):,}")
print(f"Overlap: {len(train_classes & val_classes):,} ({len(train_classes & val_classes)/len(val_classes)*100:.1f}% of val)")
print(f"Only in train: {len(train_classes - val_classes):,}")
print(f"Only in val: {len(val_classes - train_classes):,}")
EOF
```

---

## 📈 Análisis de Resultados

### Ver training history (JSON)
```bash
cd /home/skanndar/SynologyDrive/local/aplantida/ml-training

# Ver history completo
cat results/teacher_global_v1/training_history.json | python3 -m json.tool

# Extraer solo accuracies de validación
python3 << 'EOF'
import json
with open("results/teacher_global_v1/training_history.json") as f:
    data = json.load(f)
print("Val Top-1 Accuracy por epoch:")
for i, acc in enumerate(data['val_top1_acc'], 1):
    print(f"  Epoch {i}: {acc:.2f}%")
EOF
```

### Ver checkpoints guardados
```bash
# Listar checkpoints
ls -lth checkpoints/teacher_global/

# Ver información de un checkpoint específico
python3 << 'EOF'
import torch
ckpt = torch.load('checkpoints/teacher_global/last_checkpoint.pt', map_location='cpu')
print(f"Epoch: {ckpt.get('epoch', 'N/A')}")
print(f"Train Loss: {ckpt.get('train_loss', 'N/A'):.4f}")
print(f"Val Loss: {ckpt.get('val_loss', 'N/A'):.4f}")
print(f"Val Acc: {ckpt.get('val_top1_acc', 'N/A'):.2f}%")
EOF
```

### Comparar accuracy de diferentes epochs
```bash
python3 << 'EOF'
import json
with open("results/teacher_global_v1/training_history.json") as f:
    data = json.load(f)

print("\n" + "="*60)
print("  Training Progress Summary")
print("="*60)
for i in range(len(data['train_loss'])):
    print(f"\nEpoch {i+1}:")
    print(f"  Train - Loss: {data['train_loss'][i]:.4f}, Top-1: {data['train_top1_acc'][i]:.2f}%")
    print(f"  Val   - Loss: {data['val_loss'][i]:.4f}, Top-1: {data['val_top1_acc'][i]:.2f}%")
EOF
```

---

## 🧹 Mantenimiento

### Limpiar caché (liberar espacio)
```bash
# CUIDADO: Esto borrará TODAS las imágenes cacheadas
# El training tendrá que descargarlas de nuevo

# Ver tamaño actual
du -sh /media/skanndar/2TB1/aplantida-ml/image_cache

# Borrar toda la caché
rm -rf /media/skanndar/2TB1/aplantida-ml/image_cache/*

# Crear directorio de nuevo
mkdir -p /media/skanndar/2TB1/aplantida-ml/image_cache
```

### Limpiar checkpoints antiguos
```bash
cd /home/skanndar/SynologyDrive/local/aplantida/ml-training/checkpoints/teacher_global/

# Ver tamaño de checkpoints
du -sh .

# OPCIÓN 1: Borrar todos excepto los últimos 2
ls -t checkpoint_epoch_*.pt | tail -n +3 | xargs rm -f

# OPCIÓN 2: Borrar epochs específicos (ejemplo: epochs 1-5)
rm -f checkpoint_epoch_{1,2,3,4,5}.pt

# Mantener siempre: last_checkpoint.pt y best_model.pt (si existe)
```

### Limpiar logs antiguos
```bash
cd /home/skanndar/SynologyDrive/local/aplantida/ml-training

# Comprimir logs antiguos
gzip training_384_smartcrop.log.old

# Borrar logs muy antiguos (más de 30 días)
find . -name "*.log" -mtime +30 -delete
```

---

## 🚨 Troubleshooting

### Training muy lento (GPU al 2-5%)
```bash
# 1. Verificar número de workers
grep num_workers config/teacher_global.yaml

# 2. Debería ser 6 para evitar rate limiting
# Si es muy bajo, aumentar en config

# 3. Verificar que no hay rate limiting activo
grep "Rate limit" training_384_smartcrop.log | tail -20

# 4. Verificar cache hit rate
grep "cached\|downloaded" training_384_smartcrop.log | tail -50
```

### Muchos "Failed to load" o "blank images"
```bash
# Ver cuántos errores hay
grep -c "Failed to load" training_384_smartcrop.log

# Ver si es rate limiting (429)
grep "429\|Rate limit" training_384_smartcrop.log | tail -20

# Solución: Esperar que el backoff expire
# O filtrar dataset a solo imágenes cacheadas:
python3 scripts/create_stratified_split.py  # Ya filtra solo cacheadas
```

### Out of Memory (GPU)
```bash
# Reducir batch size
nano config/teacher_global.yaml

# Cambiar de:
#   batch_size: 8
# A:
#   batch_size: 4

# Guardar y reiniciar training
```

### Validation accuracy muy baja (< 1%)
```bash
# 1. Verificar que estás usando stratified split
grep train_jsonl config/teacher_global.yaml
# Debe decir: dataset_train_stratified.jsonl

# 2. Verificar overlap de clases
python3 scripts/create_stratified_split.py  # Ver output

# 3. Si sigue bajo, considerar filtrar clases con pocas imágenes
# (ver sección de configuración avanzada)
```

### Caché crece sin límite
```bash
# Verificar que el cache fix está aplicado
grep "target_size_bytes" models/streaming_dataset.py

# Debe aparecer línea con "0.90"
# Verificar tamaño configurado
grep cache_size_gb config/teacher_global.yaml

# Debería ser 220 GB (con 250GB disponibles)
```

### Smart crop no funciona (todo es center crop)
```bash
# Verificar que smart_crop está activado
grep smart_crop config/teacher_global.yaml
# Debe decir: smart_crop: true

# Verificar logs de smart crop
grep "Smart crop\|Saliency" training_384_smartcrop.log | head -20

# Test manual
python3 scripts/test_smart_crop.py --samples 3
```

---

## ⚙️ Configuración Avanzada

### Cambiar resolución de imagen
```bash
nano config/teacher_global.yaml

# Cambiar:
#   image_size: 384  # Puede ser 224, 384, 512
#
# Y en augmentation:
#   train:
#     resize: 448    # 1.16x del image_size
#     crop: 384      # Mismo que image_size
#   val:
#     resize: 448
#     center_crop: 384

# IMPORTANTE: Si cambias el tamaño, debes cambiar también el modelo:
#   model:
#     name: "vit_base_patch16_384"  # Debe coincidir con image_size
```

### Ajustar parámetros de Smart Crop
```bash
nano models/smart_crop.py

# En líneas 25-27:
#   saliency_threshold: float = 0.3  # Más bajo = más permisivo
#   min_region_ratio: float = 0.01   # Más bajo = acepta regiones pequeñas
#   use_fine_grained: bool = False   # False = Spectral Residual (mejor)
```

### Filtrar clases con pocas imágenes
```bash
# Editar create_stratified_split.py
nano scripts/create_stratified_split.py

# Añadir filtro después de cargar records (línea ~40):
#   # Filter classes with < 5 images
#   class_counts = Counter([r['class_idx'] for r in records])
#   records = [r for r in records if class_counts[r['class_idx']] >= 5]
```

### Aumentar regularización (si overfitting persiste)
```bash
nano config/teacher_global.yaml

# Cambiar:
regularization:
  dropout: 0.3          # Subir de 0.2
  mixup_alpha: 0.8      # Mantener
  cutmix_alpha: 1.0     # Mantener
  mixup_prob: 0.5       # Mantener

training:
  weight_decay: 0.05    # Subir de 0.01
  label_smoothing: 0.2  # Subir de 0.1
```

---

## 🎯 Comandos de Un Solo Paso

### Restart rápido (mantener progreso)
```bash
pkill -9 -f train_teacher; sleep 5; cd /home/skanndar/SynologyDrive/local/aplantida/ml-training && source venv/bin/activate && nohup python3 scripts/train_teacher.py --config config/teacher_global.yaml --resume checkpoints/teacher_global/last_checkpoint.pt > training_384_smartcrop.log 2>&1 & tail -f training_384_smartcrop.log | grep -E 'Epoch|loss|top1'
```

### Ver estado completo del sistema
```bash
echo "=== TRAINING STATUS ===" && ps aux | grep train_teacher | grep -v grep && echo && echo "=== GPU ===" && nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader && echo && echo "=== CACHE ===" && du -sh /media/skanndar/2TB1/aplantida-ml/image_cache && find /media/skanndar/2TB1/aplantida-ml/image_cache -name "*.jpg" | wc -l && echo && echo "=== LAST 5 CHECKPOINTS ===" && ls -lth checkpoints/teacher_global/ | head -6
```

### Ver progreso resumido
```bash
tail -200 training_384_smartcrop.log | grep -E "Epoch [0-9]+/|Train - Loss|Val   - Loss"
```

---

## 📋 Workflow Recomendado

### Flujo completo desde inicio:

```bash
# 1. Preparar entorno
cd /home/skanndar/SynologyDrive/local/aplantida/ml-training
source venv/bin/activate

# 2. Verificar/crear datasets stratified
python3 scripts/create_stratified_split.py

# 3. Test de smart crop (opcional, para verificar)
python3 scripts/test_smart_crop.py --samples 3

# 4. Iniciar training
./START_TRAINING_384.sh

# 5. Monitorear
tail -f training_384_smartcrop.log | grep -E 'Epoch|loss|top1'

# 6. Ver GPU en otra terminal
watch -n 2 nvidia-smi

# 7. Después de 5-7 epochs, evaluar si val accuracy mejora
# Si val acc < 5% después de epoch 7, considerar ajustes
```

### Reinicio tras actualizar código:

```bash
# 1. Esperar a que termine época actual
tail -f training_384_smartcrop.log

# 2. Cuando veas "Saved checkpoint_epoch_X", parar
pkill -SIGTERM -f train_teacher.py

# 3. Esperar a que guarde
sleep 10

# 4. Reiniciar
cd /home/skanndar/SynologyDrive/local/aplantida/ml-training
source venv/bin/activate
nohup python3 scripts/train_teacher.py \
    --config config/teacher_global.yaml \
    --resume checkpoints/teacher_global/last_checkpoint.pt \
    > training_384_smartcrop.log 2>&1 &

# 5. Verificar
tail -f training_384_smartcrop.log | grep -E 'Epoch|loss|top1'
```

---

## 📚 Archivos de Referencia

### Documentación
- `IMPROVEMENTS_384_SMARTCROP.md` - Detalles técnicos de mejoras 384px
- `TRAINING_ANALYSIS_EPOCH15.md` - Análisis del training anterior (224px)
- `SMART_RATE_LIMITING.md` - Sistema de rate limiting
- `SUMMARY_SESSION_20251220.md` - Resumen completo de la sesión
- `CACHE_FIX_20251219.md` - Fix de cache eviction

### Scripts útiles
- `START_TRAINING_384.sh` - Iniciar training con todas las mejoras
- `scripts/create_stratified_split.py` - Crear splits estratificados
- `scripts/test_smart_crop.py` - Visualizar smart crop
- `scripts/test_rate_limiting.py` - Test de rate limiting
- `scripts/debug_saliency.py` - Debug de detección de saliency
- `scripts/export_dataset.py` - Exportar desde MongoDB

### Configuración
- `config/teacher_global.yaml` - Config principal (384px + smart crop)
