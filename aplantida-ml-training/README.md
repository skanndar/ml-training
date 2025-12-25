# Plan de Entrenamiento ML - Aplantida

**Objetivo:** Entrenar un modelo de plantas fine-tuneado con distillation para desplegar en TensorFlow.js (offline, PWA).

**Status:** Fase 0 completada (17 dic 2025) | Fases 1-4 ejecutadas | Fase 5 (calibración + threshold) cerrada (24 dic 2025)

---

## Estado actual (diciembre 2025)

- **Fase 0 cerrada:** Pipeline de streaming con caché LRU funcionando, export completo desde Mongo y splits 80/10/10 sin leakage (`ml-training/PHASE_0_COMPLETE.md`). Resultado: 171,381 imágenes únicas, 7,120 clases en train y 92.4% de licencias permisivas.
- **Fase 1 ejecutada:** El teacher global ViT-Base (224px) completó 15 épocas (`TRAINING_ANALYSIS_EPOCH15.md`) con overfitting severo por el split aleatorio, demostrando que el código/métricas funcionan end-to-end.
- **Refactors posteriores:** El 20 de diciembre (`SUMMARY_SESSION_20251220.md`) se re-exportaron **741,533 imágenes** (365k iNat + 376k legacy), se creó `scripts/create_stratified_split.py` (train 93,761 / val 12,639 / test 11,726 con 100% de solapamiento), se validó el Smart Rate Limiting y se añadió Smart Crop por saliencia + resolución 384px (`IMPROVEMENTS_384_SMARTCROP.md`).
- **Config actual:** `config/teacher_global.yaml` usa `vit_base_patch16_384`, batch size 4, cache 220GB y `smart_crop: true`. Ejecuta `./START_TRAINING_384.sh` para la nueva corrida sobre los JSONL estratificados.
- **Pipeline regional listo:** `scripts/create_stratified_split.py --region EU_SW` + `config/teacher_regional.yaml` replican las mismas mejoras (384px + smart crop) y `./START_TRAINING_REGIONAL_384.sh` automatiza el entrenamiento del teacher B.
- **Calibración del student (Fase 5):** `scripts/temperature_scaling.py` con el checkpoint `checkpoints/student_finetune/best_model.pt` determinó una temperatura óptima **T=2.0** (ECE ➝ 0.040). El umbral recomendado para "no concluyente" es **0.66** (95.0% accuracy, 78.9% coverage) según `results/student_finetune_v1/threshold_analysis_temp.json`.
- **Export preliminar (Fase 6):** Está disponible el checkpoint cuantizado FP16 (`results/student_finetune_v1/model_fp16.pt`), el ONNX (`dist/models/student_v1_fp16_manual/student.onnx`) y el SavedModel (`dist/models/student_v1_fp16_manual/saved_model`). Falta completar la conversión a TF.js porque `tensorflowjs_converter` requiere un entorno con TensorFlow 2.13.x + `tensorflow_decision_forests` 1.5.0. Ver `EXPORT_TFJS_PWA.md` para reproducir el paso en un venv aislado.
- **Siguientes hitos:** Rerun del teacher global con las mejoras anteriores, cerrar el teacher regional y entrenar el **Teacher C (Europa Norte/East)** usando `scripts/create_stratified_split.py --region EU_NORTH,EU_EAST` + `./START_TRAINING_EU_CORE_384.sh` para cubrir el resto del continente. Una vez tengamos los tres teachers, repetir distillation/fine-tuning y volver a calibrar/exportar para TF.js.

---

## Documentos (Léelos en este orden)

1. **[ARQUITECTURA.md](./ARQUITECTURA.md)** - Sistema completo end-to-end (5 min)
2. **[DATOS_PIPELINE.md](./DATOS_PIPELINE.md)** - Extracción Mongo + CDN + augmentations (10 min)
3. **[FASES_ENTRENAMIENTO.md](./FASES_ENTRENAMIENTO.md)** - Plan de 7 fases con objetivos y comandos (15 min)
4. **[DISTILLATION_DESIGN.md](./DISTILLATION_DESIGN.md)** - Arquitectura exacta del student (10 min)
5. **[METRICAS_EVALUACION.md](./METRICAS_EVALUACION.md)** - KPIs por región y calibración (8 min)
6. **[EXPORT_TFJS_PWA.md](./EXPORT_TFJS_PWA.md)** - Export, cuantización, caché (8 min)
7. **[LICENCIAS_CHECKLIST.md](./LICENCIAS_CHECKLIST.md)** - Cumplimiento legal (5 min)
8. **[ESTRUCTURA_SCRIPTS.md](./ESTRUCTURA_SCRIPTS.md)** - Repo, CLI, config YAML (10 min)

---

## Resumen Ejecutivo

### Lo que conseguirás

- **1 modelo único** deployable en navegador (TF.js)
- **Trained by ensemble** (3 maestros → 1 alumno vía distillation + fine-tuning)
- **~9.000 especies** reconocidas con accuracy global Top-1 >75%
- **Regional optimization** para SW Europa (España +15% local accuracy)
- **Offline-first PWA** con actualización segura de modelos
- **Licencia comercial safe** (solo Apache-2.0 / MIT en producción)

### Hardware esperado

- **GPU dedicada** (RTX 3060+, A100, o similar)
- **RAM:** 32GB+ (para batch processing)
- **Almacenamiento:** 500GB SSD (dataset + checkpoints + exports)
- **Puede ejecutarse por chunks** (entrenamiento resumible)

### Timeline orientativo (NO deadlines)

- Fase 0-1: Auditoría + teacher global = 2-3 días
- Fase 2: Teacher regional = 1-2 días
- Fase 3-4: Student + distillation + fine-tuning = 3-5 días
- Fase 5-6: Calibración + export + optimizaciones = 2 días
- Fase 7: Reproducibilidad = 1 día

---

## Asumir esto es crítico

### Datos de Mongo que asumimos

```javascript
// Colección: plants
{
  _id: ObjectId,
  latinName: "Rosa canina",
  commonName: "Wild rose",
  images: [
    {
      url: "https://cdn.example.com/img/12345.jpg",
      source: "inaturalist",
      size: "original",
      license: "CC-BY-NC",
      attribution: "User123"
    }
  ],
  distribution: {
    countries: ["ES", "FR", "IT"],
    continents: ["Europe"]
  }
}

// ~9.000 especies, ~900k imágenes
// ~80-150 imágenes por especie (distribución sesgada)
```

Si el formato es diferente, ajusta los loaders en `DATOS_PIPELINE.md`.

### Teachers que usaremos

1. **Maestro A (Global):** ViT-Base o ResNet-50 preentrenado en ImageNet (MIT/Apache-2.0)
   - Para diversidad general

2. **Maestro B (Regional):** Fine-tuned en GBIF SW Europa (si existe modelo CC0/Apache)
   - Si no existe, entrenaremos uno usando Odoo data

3. **Maestro C (Opcional):** BioCLIP o ResNet fine-tuned botánico (evaluar licencia)

### Licencias - regla de oro

```
Experimental (I+D):    Puedes usar NC/GPL/Cualquiera
Producción (Web PWA):  SOLO Apache-2.0, MIT, BSD, CC0

Fine-tuning no elimina restricciones originales.
→ Si teacher es NC, el student hereda NC.
→ Solución: usar solo teachers permisivos en producción.
```

---

## Decisiones arquitectónicas clave

### ¿Por qué distillation + fine-tuning?

- **Distillation:** El student aprende patrones de 3 maestros sin tener que cargarlos todos
- **Fine-tuning:** Luego lo ajustamos con tus datos reales de Odoo para máxima precisión local
- **Resultado:** 1 modelo ~100MB deployable, rendimiento ≈ ensemble

### ¿Por qué teacher regional?

- Especies de SW Europa tienen características visuales distintas
- Datos GBIF regionalizados + observaciones iNaturalist locales
- Top-1 accuracy en España: +15% respecto a modelo global puro

### ¿Por qué TF.js en navegador?

- PWA offline-first (crítico para aplicaciones agrícolas/rurales)
- Cero latencia (sin round-trip a servidor)
- Usuario data privacy (imágenes no salen del navegador)
- Model caching + versioning via Service Worker

---

## Cómo usar este plan

1. **Lee ARQUITECTURA.md primero** - entiende el flujo general
2. **Prepara DATOS_PIPELINE.md** - valida datos y escala
3. **Ejecuta FASES_ENTRENAMIENTO.md en orden** - cada fase tiene checklist
4. **Itera:** Si una métrica falla, vuelve a la fase anterior (checkpoints guardados)
5. **Deploy:** EXPORT_TFJS_PWA.md tiene instrucciones PWA + actualización

---

## Riesgos principales & mitigaciones

| Riesgo | Impacto | Mitigación |
|--------|--------|-----------|
| Imágenes corruptas en CDN | Entrenamiento falla | Validación EXIF + hash en Fase 0 |
| Desbalanceo extremo (80 vs 15k imágenes) | Overfitting en clases mayores | Sampling + stratified splits |
| GPU memory overflow | Out of memory | Batch chunking + gradient accumulation |
| Modelo no convergea (val loss aumenta) | Modelo inútil | Learning rate schedule, early stopping |
| Confusión entre sp. similares (ej. Rosa vs Rubus) | Baja precision local | Análisis de confusiones + re-entrenamiento |
| Service Worker cache inválido | Usuarios stuck con modelo viejo | Versioning + fallback a modelo anterior |

---

## Criterios de éxito (Fase 7)

- [ ] Global Top-1 >= 75%
- [ ] Global Top-5 >= 90%
- [ ] España Top-1 >= 88% (regional boost)
- [ ] ECE (Expected Calibration Error) < 0.08
- [ ] Tasa "no conclusive" < 5% (umbral configurable)
- [ ] TF.js model <= 150MB (cuantizado)
- [ ] Inferencia en navegador < 2s (GPU) / < 5s (CPU)
- [ ] Todos los scripts reproducibles (seed fijo)

---

## Próximos pasos

1. Genera/valida los JSONL estratificados (`scripts/create_stratified_split.py`) y revisa que el caché contenga todas las URLs del split.
2. Lanza `./START_TRAINING_384.sh` (o `python scripts/train_teacher.py --config config/teacher_global.yaml`) para la nueva corrida con smart crop + 384px.
3. Monitorea `training_384_smartcrop.log`, TensorBoard (`checkpoints/teacher_global/logs`) y documenta métricas/insights en `results/teacher_global_v1/`.
4. Con el teacher global estable, inicia la Fase 2 (teacher regional) reutilizando los mismos filtros/licencias.
5. Repite `scripts/create_stratified_split.py --region EU_NORTH,EU_EAST --output-prefix ./data/dataset_eu_core` y lanza `./START_TRAINING_EU_CORE_384.sh` para entrenar el Teacher C (resto de Europa) antes de pasar a distillation.

¿Preguntas? Cada doc tiene sección "Troubleshooting".
