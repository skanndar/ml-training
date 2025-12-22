# Pipeline de Datos: Mongo + CDN + Augmentations

## Fase 0: Auditoría Dataset

### 0.1 - Validar estructura Mongo

```python
# scripts/audit_dataset.py

from pymongo import MongoClient
import requests
from PIL import Image
from io import BytesIO

client = MongoClient("mongodb://localhost:27017")
db = client["aplantida_db"]
plants = db.plants

# Estadísticas
total_plants = plants.count_documents({})
print(f"Total species: {total_plants}")

# Distribución de imágenes
image_stats = plants.aggregate([
    {"$group": {
        "_id": None,
        "total_images": {"$sum": {"$size": "$images"}},
        "avg_per_species": {"$avg": {"$size": "$images"}},
        "min": {"$min": {"$size": "$images"}},
        "max": {"$max": {"$size": "$images"}}
    }}
])

for stat in image_stats:
    print(f"Images: {stat}")

# Verificar campos requeridos
missing_fields = plants.count_documents({
    "$or": [
        {"latinName": {"$exists": False}},
        {"images": {"$exists": False}},
    ]
})
print(f"Records missing required fields: {missing_fields}")

# Licencias presentes
licenses = plants.aggregate([
    {"$unwind": "$images"},
    {"$group": {"_id": "$images.license", "count": {"$sum": 1}}}
])
print(f"Licenses distribution: {list(licenses)}")

# Fuentes de imágenes
sources = plants.aggregate([
    {"$unwind": "$images"},
    {"$group": {"_id": "$images.source", "count": {"$sum": 1}}}
])
print(f"Image sources: {list(sources)}")
```

### 0.2 - Validar URLs y descargas

```python
# scripts/validate_image_urls.py

import concurrent.futures
import hashlib
from pathlib import Path

def validate_url(url, timeout=10):
    try:
        resp = requests.head(url, timeout=timeout, allow_redirects=True)
        if resp.status_code == 200:
            # Validar Content-Type
            content_type = resp.headers.get('Content-Type', '')
            if 'image' in content_type:
                return {"url": url, "status": "ok", "size": resp.headers.get('Content-Length')}
            else:
                return {"url": url, "status": "not_image"}
        else:
            return {"url": url, "status": f"http_{resp.status_code}"}
    except Exception as e:
        return {"url": url, "status": f"error_{str(e)[:20]}"}

# Muestreo aleatorio de 1000 URLs
sample_urls = plants.aggregate([
    {"$sample": {"size": 1000}},
    {"$unwind": "$images"},
    {"$project": {"url": "$images.url"}}
])

urls_list = [doc["url"] for doc in sample_urls]

# Validar en paralelo
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(validate_url, urls_list))

failed = [r for r in results if r["status"] != "ok"]
print(f"URL validation: {len(urls_list) - len(failed)}/{len(urls_list)} ok")
if failed:
    print(f"Failed: {failed[:10]}")  # Primeros 10 fallos

# Estadísticas de tamaño
sizes = [int(r.get("size", 0)) for r in results if r["status"] == "ok"]
print(f"Average image size: {sum(sizes) / len(sizes) / 1024 / 1024:.1f} MB")
```

---

## Fase 1: Export y Streaming desde Mongo

### 1.1 - Export a JSONL

```python
# scripts/export_dataset.py

"""
Exporta datos desde Mongo a JSONL (línea por imagen).
Incluye validación de licencias y metadatos geográficos.
"""

import json
from pymongo import MongoClient
from pathlib import Path

PERMISSIVE_LICENSES = [
    "CC0", "CC-BY", "CC-BY-SA",
    "public-domain",
    "No copyright",
]

def export_to_jsonl(output_dir: str, min_images_per_species: int = 5):
    """
    Exporta a JSONL.
    Cada línea = {"plant_id", "latin_name", "image_url", "split", "region", ...}
    """
    client = MongoClient("mongodb://localhost:27017")
    db = client["aplantida_db"]
    plants = db.plants

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    records = []

    for plant in plants.find({}):
        if len(plant.get("images", [])) < min_images_per_species:
            continue

        latin_name = plant.get("latinName", "UNKNOWN")
        plant_id = str(plant["_id"])

        for img in plant.get("images", []):
            # Filtrar licencias no permisivas
            license_str = img.get("license", "")
            if license_str and not any(perm in license_str for perm in PERMISSIVE_LICENSES):
                continue

            record = {
                "plant_id": plant_id,
                "latin_name": latin_name,
                "common_name": plant.get("commonName", ""),
                "image_url": img.get("url", ""),
                "image_source": img.get("source", ""),
                "license": license_str,
                "attribution": img.get("attribution", ""),
                "region": get_region(plant),  # función abajo
                "country": extract_country(plant),  # función abajo
            }
            records.append(record)

    print(f"Total valid records: {len(records)}")

    # Exportar a JSONL
    with open(output_path / "dataset_raw.jsonl", "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"Exported to {output_path}/dataset_raw.jsonl")

def get_region(plant: dict) -> str:
    """Mapeo geográfico simplificado."""
    distribution = plant.get("distribution", {})
    countries = set(distribution.get("countries", []))

    # Definir regiones
    EU_SW = {"ES", "PT", "FR", "IT"}
    EU = {"DE", "UK", "NL", "BE", "AT", "CH", "SE", "NO"}

    if countries & EU_SW:
        return "EU_SW"
    elif countries & EU:
        return "EU_NORTH"
    elif "NA" in distribution.get("continents", []):
        return "AMERICAS"
    else:
        return "OTHER"

def extract_country(plant: dict) -> str:
    countries = plant.get("distribution", {}).get("countries", [])
    return countries[0] if countries else "UNKNOWN"
```

### 1.2 - Data Loader con streaming

```python
# scripts/data_loader.py

"""
Loader eficiente para imágenes desde CDN.
Implementa caché local, validación, y augmentations.
"""

import tensorflow as tf
import requests
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
import hashlib

class PlantDataLoader:
    def __init__(self, jsonl_path: str, cache_dir: str = "./data_cache"):
        self.jsonl_path = jsonl_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._load_records()

    def _load_records(self):
        """Carga JSONL en memoria."""
        self.records = []
        with open(self.jsonl_path) as f:
            for line in f:
                self.records.append(json.loads(line))
        print(f"Loaded {len(self.records)} records")

    def download_image(self, url: str, plant_id: str) -> bool:
        """Descarga imagen con caché."""
        cache_path = self.cache_dir / f"{plant_id}_{hashlib.md5(url.encode()).hexdigest()}.jpg"

        if cache_path.exists():
            return True

        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                cache_path.write_bytes(resp.content)
                return True
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return False

        return False

    def get_image_tensor(self, image_path: str, size: tuple = (224, 224)) -> tf.Tensor:
        """Lee imagen local y retorna tensor normalizado."""
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, size)
        img = img / 255.0  # Normalizar a [0, 1]
        return img

    def create_tf_dataset(self, split: str = "train", batch_size: int = 32):
        """
        Crea tf.data.Dataset desde JSONL.
        Soporta augmentations y sampling balanceado.
        """
        # Filtrar por split (asumimos ya está en JSONL)
        records = [r for r in self.records if r.get("split") == split]

        # Descargar imágenes en paralelo
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(self.download_image, r["image_url"], r["plant_id"])
                for r in records
            ]
            results = [f.result() for f in futures]

        valid_records = [r for r, success in zip(records, results) if success]
        print(f"Valid records for {split}: {len(valid_records)}/{len(records)}")

        # Crear dataset
        dataset = tf.data.Dataset.from_tensor_slices(valid_records)

        if split == "train":
            dataset = dataset.shuffle(buffer_size=len(valid_records))
            # Augmentation para train
            dataset = dataset.map(
                lambda r: self._augmented_record(r),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            # Sin augmentation para val/test
            dataset = dataset.map(
                lambda r: self._plain_record(r),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def _augmented_record(self, record: dict):
        """Aplica augmentations de entrenamiento."""
        # Implementar dentro del graph de TF
        pass

    def _plain_record(self, record: dict):
        """Sin augmentation."""
        pass
```

---

## Fase 2: Balanceo y Split

### 2.1 - Estratified split

```python
# scripts/split_dataset.py

"""
Split train/val/test evitando data leakage.
Estrategias:
- Por especie (mantener ratio)
- Por región
- Por individuo (si existe en metadata)
"""

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import pandas as pd

def create_splits(jsonl_path: str, output_dir: str, seed: int = 42):
    """
    Crea splits 80/10/10 con stratificación.
    """
    df = pd.read_json(jsonl_path, lines=True)

    print(f"Dataset shape: {df.shape}")
    print(f"\nClass distribution:\n{df['latin_name'].value_counts().head(20)}")
    print(f"\nRegion distribution:\n{df['region'].value_counts()}")

    # Estratificación por especies + región
    df["strata"] = df["latin_name"] + "_" + df["region"]

    # Split 80/20
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    for train_idx, temp_idx in sss.split(df, df["strata"]):
        train_df = df.iloc[train_idx]
        temp_df = df.iloc[temp_idx]

    # Split 50/50 del 20% → 10% val, 10% test
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    for val_idx, test_idx in sss2.split(temp_df, temp_df["strata"]):
        val_df = temp_df.iloc[val_idx]
        test_df = temp_df.iloc[test_idx]

    # Agregar split column
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Guardar
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_df = pd.concat([train_df, val_df, test_df])
    final_df.to_json(output_dir / "dataset_splits.jsonl", orient="records", lines=True)

    print(f"\nTrain: {len(train_df)}")
    print(f"Val: {len(val_df)}")
    print(f"Test: {len(test_df)}")

    # Verificar no hay leakage por individuo (si existe campo)
    if "individual_id" in final_df.columns:
        train_indiv = set(final_df[final_df["split"] == "train"]["individual_id"])
        val_indiv = set(final_df[final_df["split"] == "val"]["individual_id"])
        leakage = len(train_indiv & val_indiv)
        print(f"Individual leakage: {leakage}")
```

---

## Fase 3: Augmentations

### 3.1 - Definir augmentation pipeline

```python
# scripts/augmentations.py

"""
Augmentations específicas para datos botánicos.
Balance: máximo beneficio sin perder identidad.
"""

import albumentations as A
import cv2

def get_train_augmentation(image_size: int = 224):
    """
    Augmentations para training.
    Conservador: preserva características diagnósticas.
    """
    return A.Compose([
        A.Resize(image_size, image_size),

        # Rotación suave (plantas pueden estar inclinadas)
        A.Rotate(limit=30, p=0.5),

        # Flip horizontal (OK para plantas)
        A.HorizontalFlip(p=0.5),

        # Color jitter (ej. diferentes iluminaciones, épocas del año)
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.7
        ),

        # Slight zoom (planta más cerca/lejos)
        A.Perspective(scale=(0.05, 0.1), p=0.3),

        # Simular sombras parciales
        A.CoarseDropout(max_holes=1, max_height=20, max_width=20, p=0.3),

        # Ruido ligero (fotos en mala luz)
        A.GaussNoise(p=0.2),

        # Normalizar
        A.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
        A.pytorch.ToTensorV2(),
    ], bbox_params=None, keypoint_params=None)

def get_val_augmentation(image_size: int = 224):
    """Sin augmentation, solo resize + normalize."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
        A.pytorch.ToTensorV2(),
    ])
```

---

## Resumen: Comandos para Fase 0-3

```bash
# 0. Auditar dataset
python scripts/audit_dataset.py
python scripts/validate_image_urls.py

# 1. Exportar desde Mongo
python scripts/export_dataset.py --output_dir ./data

# 2. Split train/val/test
python scripts/split_dataset.py \
    --jsonl_path ./data/dataset_raw.jsonl \
    --output_dir ./data

# 3. Listo para entrenamiento
ls -lh ./data/
# → dataset_raw.jsonl (todos registros)
# → dataset_splits.jsonl (con split column)
# → data_cache/ (imágenes descargadas)
```

Antes de continuar a FASES_ENTRENAMIENTO.md, verifica:
- [ ] Todas las URLs válidas (§ 0.2)
- [ ] Split sin leakage (§ 2.1)
- [ ] Augmentations no dañan identidad (§ 3.1)
