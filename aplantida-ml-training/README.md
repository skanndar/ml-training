# Plan de Entrenamiento ML - Aplantida

**Objetivo:** Entrenar un modelo de plantas fine-tuneado con distillation para desplegar en TensorFlow.js (offline, PWA).

**Status:** Diseño completo | Listo para ejecución

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

1. Valida el formato de datos en Mongo (DATOS_PIPELINE.md § 0)
2. Prepara hardware (GPU, almacenamiento)
3. Clona repo + crea venv Python (ESTRUCTURA_SCRIPTS.md)
4. Comienza Fase 0 (auditoría dataset)

¿Preguntas? Cada doc tiene sección "Troubleshooting".
