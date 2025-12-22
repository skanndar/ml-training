# Solución Final al Problema de Rate Limiting

## Problema

iNaturalist está bloqueando las descargas (429 Too Many Requests), causando:
- Miles de imágenes "blank" (negras)
- Accuracy contaminada
- Training lento (intentos fallidos de descarga)

## Solución: Training Solo con Caché Existente

### Opción 1: Usar dataset_raw.jsonl (340k imágenes disponibles)

Tienes 340,749 imágenes en `dataset_raw.jsonl` (más del doble de las que estás usando).

**Pasos:**

1. Filtrar dataset_raw.jsonl para quedarnos solo con imágenes cacheadas
2. Partir en train/val/test
3. Entrenar con eso

```bash
cd /home/skanndar/SynologyDrive/local/aplantida/ml-training

# Script para filtrar y partir dataset_raw
python3 << 'EOF'
import hashlib
import json
import random
from pathlib import Path

def url_to_filename(url):
    return hashlib.md5(url.encode()).hexdigest() + ".jpg"

# Cargar caché
cache_dir = Path("/media/skanndar/2TB1/aplantida-ml/image_cache")
cached = {f.name for f in cache_dir.glob("*.jpg")}
print(f"Cached images: {len(cached):,}")

# Filtrar dataset_raw
records = []
with open("data/dataset_raw.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        if url_to_filename(rec['image_url']) in cached:
            records.append(rec)

print(f"Found {len(records):,} cached records in dataset_raw")

# Shuffle y partir
random.shuffle(records)
train_size = int(len(records) * 0.8)
val_size = int(len(records) * 0.1)

train = records[:train_size]
val = records[train_size:train_size + val_size]
test = records[train_size + val_size:]

# Guardar
for name, data in [('train', train), ('val', val), ('test', test)]:
    with open(f"data/dataset_{name}_from_raw_cached.jsonl", "w") as f:
        for rec in data:
            f.write(json.dumps(rec) + "\n")
    print(f"{name}: {len(data):,} images")
EOF
```

### Opción 2: Deshabilitar Descargas en el Código

Modificar `streaming_dataset.py` para que SI una imagen no está en caché, la SKIP en lugar de intentar descargar:

```python
# En línea ~398 de streaming_dataset.py, cambiar:

else:
    # Download
    img = None
    image_bytes = self.downloader.download(url)
    ...

# POR:

else:
    # SKIP - No download, cache-only mode
    img = None
```

Pero esto causará que el dataset se salte imágenes, lo cual PyTorch no maneja bien.

### Opción 3: MEJOR - Usar MongoDB

Si tienes 900k imágenes en MongoDB, puedes:

1. Exportar JSONL desde MongoDB con las imágenes que SÍ tienes
2. Filtrar por las que están en caché
3. Entrenar con eso

¿Puedes darme las credenciales de MongoDB para verificar cuántas imágenes hay realmente?

```bash
mongosh --username <user> --password <pass> --authenticationDatabase admin RehabDatos
```

O si está sin auth:
```bash
mongosh RehabDatos --eval "db.plants.countDocuments()"
```

## Recomendación Inmediata

PARAR el training actual y usar dataset_raw filtrado:

```bash
# 1. Parar training
pkill -9 -f train_teacher.py

# 2. Filtrar dataset_raw (ejecutar el script de Python arriba)

# 3. Actualizar config
nano config/teacher_global.yaml
# Cambiar a:
#   train_jsonl: "./data/dataset_train_from_raw_cached.jsonl"
#   val_jsonl: "./data/dataset_val_from_raw_cached.jsonl"

# 4. Reiniciar
source venv/bin/activate
nohup python3 scripts/train_teacher.py --config config/teacher_global.yaml > training.log 2>&1 &
```

## Sobre las 900k Imágenes

Para escalar a 900k imágenes:

1. **Pre-descargar todo**: Antes de entrenar, descargar las 900k imágenes en batches pequeños (100-200/hora) durante días para evitar rate limit

2. **Usar S3 propio**: Subir las imágenes a tu propio S3/Cloudflare R2 para no depender de iNaturalist

3. **Dataset local**: Descargar todo a disco local (900k × 1.5MB = ~1.3TB)

¿Cuál opción prefieres que ejecutemos ahora?
