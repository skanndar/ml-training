# Arquitectura End-to-End

## Diagrama del sistema completo

```
┌─────────────────────────────────────────────────────────────────┐
│                      FASE 0-1: DATOS                             │
└─────────────────────────────────────────────────────────────────┘
         ↓
    MongoDB (9k species, 900k images)
         ↓
    CDN URLs (inaturalist, gbif, perenual, etc)
         ↓
    export_dataset.py → JSONL + image streaming
         ↓
    Validación: corruptas, tamaños, licencias
         ↓
    train_val_test split (80/10/10)
    └─→ stratified por clase + por región

┌─────────────────────────────────────────────────────────────────┐
│                    FASE 1-2: TEACHERS                            │
└─────────────────────────────────────────────────────────────────┘
         ↓
    Teacher A: ViT-Base (ImageNet)
    └─→ train_teacher_global.py
        (fine-tune global, eval todas las especies)
        (split estratificado + smart crop 384px)
         ↓
    Teacher B: Regional SW Europa
    └─→ train_teacher_regional.py
        (fine-tune con EU_SW estratificado + smart crop 384px)
        (usa create_stratified_split.py --region EU_SW)
        (eval España específicamente)
         ↓
    Teacher C: Opcional (BioCLIP o similar)
    └─→ train_teacher_optional.py
        (ej: EU_NORTH + EU_EAST vía create_stratified_split.py --region EU_NORTH,EU_EAST)
        (balancea especies centro/norte de Europa)

    Outputs: 3x checkpoint + logits precomputados

┌─────────────────────────────────────────────────────────────────┐
│                    FASE 3-4: DISTILLATION                        │
└─────────────────────────────────────────────────────────────────┘
         ↓
    Student Model: MobileNetV2 (~15MB base)
         ↓
    Generar soft labels (logits teachers en train set)
    └─→ generate_soft_labels.py (on-the-fly o precomputed)
         ↓
    Distillation Loss:
    ├─ KL divergence (teacher logits vs student logits)
    ├─ Cross-entropy (soft labels + hard labels balance)
    └─ Temperatura: T=3.0 (inicialmente, ajustable)
         ↓
    train_student_distill.py
    └─→ outputa student con pesos destilados
         ↓
    Fine-tuning con labels reales:
    └─→ train_student_finetune.py
        (LR más bajo, data augmentation agresiva)

    Output: student_final.h5 + student_final_onnx

┌─────────────────────────────────────────────────────────────────┐
│                 FASE 5-6: EVALUACIÓN                             │
└─────────────────────────────────────────────────────────────────┘
         ↓
    eval_metrics.py
    ├─ Top-1 / Top-5 global
    ├─ Top-1 / Top-5 por región (España vs resto)
    ├─ ECE (Expected Calibration Error)
    ├─ Confusion matrix (Top-20 clases confundidas)
    └─ Per-class precision (clases imbalanceadas)
         ↓
    calibration.py
    └─ Ajustar umbral "no conclusive"
       (si max(probs) < threshold → "No se identifica")
         ↓
    comparar_vs_ensemble.py
    └─ ¿Student ≈ Teachers? (KL divergence, accuracy gap)

┌─────────────────────────────────────────────────────────────────┐
│            FASE 6: OPTIMIZACIÓN & EXPORT TF.JS                   │
└─────────────────────────────────────────────────────────────────┘
         ↓
    Cuantización:
    ├─ FP16 (half-precision) → ~7.5MB
    ├─ INT8 (post-training) → ~4MB
    └─ Evaluamos trade-off accuracy vs size
         ↓
    export_to_tfjs.py
    ├─ Keras H5 → SavedModel
    ├─ SavedModel → TF.js (model.json + *.bin shards)
    └─ Validar equivalencia: python vs browser
         ↓
    Output: dist/models/student_v1.0/
            ├─ model.json (metadatos)
            ├─ group1-shard1of3.bin
            ├─ group1-shard2of3.bin
            └─ group1-shard3of3.bin

┌─────────────────────────────────────────────────────────────────┐
│         FASE 7: DESPLIEGUE PWA (FRONT-END)                      │
└─────────────────────────────────────────────────────────────────┘
         ↓
    Service Worker (Workbox)
    ├─ Precaché del model.json + shards
    ├─ Versionado: v1.0, v1.1, etc.
    └─ Política: cache-first + background sync
         ↓
    PlantRecognition.js (ya existe)
    └─ reemplaza mobilenet por student custom
        ├─ tf.loadLayersModel('indexeddb://aplantida-model-v1.0')
        ├─ const predictions = model.predict(preprocessed_img)
        └─ mapeo a plant IDs vía lookup table
         ↓
    Model update strategy:
    ├─ Nuevo model → genera v1.1/model.json
    ├─ Service Worker detecta cambio
    ├─ Background: descarga v1.1 mientras usuario sigue con v1.0
    ├─ Post-install: avisa usuario ("Actualización disponible")
    └─ User accept → switch a v1.1 + reload
         ↓
    Output: PWA con modelo embedido
            (offline mode funcional sin servidor)

┌─────────────────────────────────────────────────────────────────┐
│              FEEDBACK LOOP (CONTINUOUSLY)                        │
└─────────────────────────────────────────────────────────────────┘
         ↓
    Cada recognition guardada en DB:
    {
      user_id, image_url, model_version,
      predicted_species, confidence,
      user_feedback (correct/incorrect/other)
    }
         ↓
    collect_feedback.py
    └─ Ejecutar mensualmente
       (identifica false positives, clases confundidas)
         ↓
    re-entrenamiento Fase 4 (fine-tuning)
    └─ Con nuevas muestras + datos user feedback
```

---

## Flujos de datos principales

### 1. Export dataset desde Mongo

```python
# mongo_export.py pseudocode
for species in plants.find({}):
    for image in species.images:
        if image.license in PERMISSIVE_LICENSES:
            download_image(image.url)
            record_metadata(
                plant_id, latin_name, common_name,
                image_url, image_source, license,
                lat/lng (if available), country
            )
            # Output: train.jsonl, val.jsonl, test.jsonl
```

### 2. Inferencia en navegador

```javascript
// PlantRecognition.js (actualizado)
async function recognizeWithStudent(imageElement) {
  // 1. Cargar modelo desde IndexedDB (caché)
  const model = await tf.loadLayersModel('indexeddb://aplantida-student-v1.0');

  // 2. Preprocesar imagen
  const tensor = tf.image.resizeBilinear(
    tf.browser.fromPixels(imageElement),
    [224, 224]  // o tamaño del student
  );
  const normalized = tensor.div(255.0);

  // 3. Inferencia
  const logits = model.predict(normalized);
  const probs = tf.softmax(logits);

  // 4. Top-K
  const top5 = tf.topk(probs, 5);

  // 5. Mapeo a plant IDs (lookup table en IndexedDB)
  const results = mapToSpecies(top5.indices, top5.values);

  // 6. Aplicar umbral "no conclusive"
  if (results[0].confidence < CONFIDENCE_THRESHOLD) {
    return { error: "No conclusive" };
  }

  return results;
}
```

### 3. Splits estratificados (global / EU_SW)

```bash
# Global
python scripts/create_stratified_split.py \
  --input ./data/dataset_raw.jsonl \
  --output-prefix ./data/dataset \
  --cache-dir ./data/image_cache

# Regional SW Europa (mismas mitigaciones contra overfitting)
python scripts/create_stratified_split.py \
  --input ./data/dataset_raw.jsonl \
  --output-prefix ./data/dataset_eu_sw \
  --region EU_SW \
  --cache-dir ./data/image_cache

# Salida: *_train/val/test_stratified.jsonl + *_stratified_stats.json
# Consumidos por train_teacher_global.py y train_teacher_regional.py
```


---

## Componentes clave

### A. Dataset & Preprocessing

- **Fuente:** Mongo + CDN streaming
- **Validación:** EXIF, tamaño, corrupción
- **Augmentation:** Rotación, flip, color jitter, cutmix (en train)
- **Output:** TFRecord o LMDB (optimizado para lectura paralela)

### B. Teachers

- **Arquitectura:** ViT-Base o ResNet-50
- **Pretrained:** ImageNet pesos
- **Fine-tuning:** 10-20 épocas con LR baja (1e-5)
- **Regularización:** Dropout, weight decay
- **Output:** Checkpoints + logits en train/val sets

### C. Student

- **Arquitectura:** MobileNetV2 (α=1.0)
- **Razón:** Balance eficiencia/accuracy, TF.js optimizado
- **Distillation:** KL divergence + cross-entropy
- **Fine-tuning:** Fine-grained classification (aggressive augmentation)
- **Output:** ONNX + Keras H5 + TF.js

### D. Evaluación

- **Offline metrics:** Top-1, Top-5, ECE, confusion matrix
- **Per-region:** España vs resto de Europa vs mundo
- **Calibration:** Temperature scaling si ECE > 0.1
- **Online tracking:** Recognition history en DB (user feedback loop)

### E. Deployment

- **Service Worker:** Precaché de assets + modelo
- **Model versioning:** IndexedDB con manifest
- **Update mechanism:** Background sync + user notification
- **Fallback:** Si modelo no carga, usar PlantNet API

---

## Tecnología stack

| Componente | Tech | Razón |
|-----------|------|-------|
| **Data** | MongoDB + Cloudinary CDN | Ya existe |
| **Training** | PyTorch + HuggingFace Transformers | Distillation nativa |
| **Evaluation** | scikit-learn + matplotlib | Standard |
| **Export** | TensorFlow Lite + TF.js converter | Compatibility |
| **Frontend** | React + TF.js | Ya existe |
| **Caching** | Workbox + IndexedDB | PWA standard |
| **Orchestration** | Python + Bash scripts | Reproducible |

---

## Supuestos críticos

1. **Mongo dataset es consistente** (latinName único por especie)
2. **CDN URLs son públicas y estables**
3. **Licencias de imágenes están documentadas**
4. **~9.000 especies es número final** (si crece, re-entrenar)
5. **GPU memory >= 8GB** para batch size razonable
6. **Usuarios tienen browser >= ES2020** (TF.js requiere)

Si alguno falla, documentamos en sección "Troubleshooting" de cada fase.
