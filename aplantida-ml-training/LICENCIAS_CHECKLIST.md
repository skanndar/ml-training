# Licencias: Checklist de Cumplimiento

## Contexto Legal

**Aplantida Web:** Actualmente gratuita, potencialmente monetizada con anuncios en futuro.

**Restricción:** Producción debe usar SOLO licencias permisivas (no restrictivas como GPL, NC).

**Razón:** Modelos fine-tuneados heredan restricciones del original (obra derivada).

---

## Estado actual (diciembre 2025)

- **Fase 0 documentada (17 dic 2025):** `dataset_train.jsonl` tiene 136,312 muestras (7,120 clases) con 92.4% de licencias permisivas (CC0/CC-BY/CC-BY-SA). Ver `../PHASE_0_COMPLETE.md`.
- **Refactor Fase 1:** `scripts/create_stratified_split.py` genera `dataset_{train,val,test}_stratified.jsonl` (93,761 / 12,639 / 11,726) solo con imágenes cacheadas. Cada registro mantiene el campo `license` para trazabilidad.
- **Nueva exportación (20 dic 2025):** 741,533 imágenes disponibles (365,025 iNaturalist + 376,481 legacy). Las fuentes legacy (propias) son permisivas; las de iNaturalist siguen requiriendo filtro por imagen (`SUMMARY_SESSION_20251220.md`).
- **Teachers en uso:** `vit_base_patch16_384` vía `timm` (Apache-2.0) y MobileNetV2 (TensorFlow, Apache-2.0) para distillation. Soft labels se generarán solo desde teachers permisivos.
- **Objetivo pendiente:** elevar el split de producción a ≥95% licencias permisivas revisando/filtrando las muestras CC-BY-NC restantes del train estratificado antes de distillation/export.

---

## Análisis por Componente

### 1. Teachers (Modelos)

#### Teacher A: ViT-Base (ImageNet)

**Uso actual (dic 2025):** `vit_base_patch16_384` desde `timm` (Apache-2.0). Configurado en `ml-training/config/teacher_global.yaml` y lanzado vía `START_TRAINING_384.sh`.

**Opción 1: Timm (PyTorch Image Models)**
```
Modelo: vision_transformer (ViT-Base)
Licencia: Apache-2.0 ✓
Pesos: ImageNet pretraining (public)
Status: SEGURO PARA PRODUCCIÓN
```

**Opción 2: Transformers HuggingFace**
```
Modelo: google/vit-base-patch16-224-in21k
Licencia: Apache-2.0 ✓
Pesos: De Google, permisivos
Status: SEGURO
```

**Evitar:**
- ModelsGenesis (NC - no comercial)
- BioCLIP (revisar licencia)

#### Teacher B: Regional (SW Europa)

**Opción 1: Fine-tune GBIF data**
```
GBIF dataset: CC0 (dominio público) ✓
iNaturalist images: Múltiples (revisar cada imagen) ⚠
Resultado: Licencia heredada = mínima restrictiva
Status: REQUIRES REVIEW
```

**Action Plan:**
```python
# Filtrar solo imágenes CC0 o CC-BY
PERMISSIVE_LICENSES = {
    'CC0', 'public domain',
    'CC-BY', 'CC-BY-SA'
}

# En export_dataset.py
for image in plant.images:
    license = image.get('license', '')
    if not any(lic in license for lic in PERMISSIVE_LICENSES):
        skip_image()  # No usar en training
```

#### Teacher C: Opcional

**NO usar:**
- BioCLIP (si licencia es restrictiva)
- Modelos con NC/ND restricciones
- Pesos no liberados públicamente

**OK usar:**
- ResNet50 ImageNet (PyTorch, Apache)
- Inception (TensorFlow, Apache)
- MobileNet (TensorFlow, Apache)

---

### 2. Student Model

**MobileNetV2 Base:**
```
Origen: TensorFlow (Apache-2.0) ✓
Pesos: ImageNet pretraining (public)
Fine-tuning: Heredada de teachers
CONCLUSIÓN: SEGURO si teachers son seguros
```

**Truth:**
Si teachers son:
- Apache-2.0 → Student hereda Apache-2.0 ✓
- MIT → Student hereda MIT ✓
- CC-BY → Student hereda CC-BY ⚠ (requiere atribución)
- NC (no-commercial) → Student hereda NC ✗ (PROBLEMA)

---

### 3. Dataset de Imágenes

#### Snapshot actual

- `dataset_raw.jsonl` (17 dic) → 340,749 imágenes, 161,791 LIC permisivas (47.5%), 178,958 restringidas (`PHASE_0_COMPLETE.md`).
- `dataset_train.jsonl` → 136,312 muestras (92.4% permisivas) usadas en la primera corrida ViT.
- `dataset_{train,val,test}_stratified.jsonl` → 93,761 / 12,639 / 11,726 registros cacheados, cada línea incluye `license` para filtrar antes de distillation.
- Nueva exportación 741,533 imágenes (20 dic) → combina 365k iNaturalist + 376k legacy (propias). Licencia final dependerá de cuántas iNat CC-BY-NC se descarten.

#### iNaturalist

```json
{
  "source": "iNaturalist",
  "license_distribution": {
    "CC-BY-NC": "45%",    // ✗ NO usar en producción
    "CC-BY": "30%",       // ✓ OK (requiere atribución)
    "CC-BY-SA": "20%",    // ✓ OK
    "CC0": "5%"           // ✓ OK
  }
}
```

**Solución:**
```python
# En scripts/export_dataset.py

PRODUCTION_LICENSES = {'CC0', 'CC-BY', 'CC-BY-SA'}
EXPERIMENTAL_LICENSES = PRODUCTION_LICENSES | {'CC-BY-NC', 'GPL'}

def get_license_tier(license_str):
    if any(lic in license_str for lic in PRODUCTION_LICENSES):
        return 'production'
    elif any(lic in license_str for lic in EXPERIMENTAL_LICENSES):
        return 'experimental'
    else:
        return 'restricted'

# Exportar dos datasets
for image in images:
    tier = get_license_tier(image.license)

    if tier == 'production':
        write_to_production_dataset(image)
    elif tier == 'experimental':
        write_to_experimental_dataset(image)
    else:
        log_warning(f"Skipped {image.url}: restricted license")
```

#### GBIF

```
License: CC0 (public domain) ✓
Issue: Metadata might require attribution
Solution: Include GBIF in footer if distributing dataset
```

#### Perenual

```
License: Check per image ⚠
Many: CC-BY-NC ✗
Action: Filter CC-BY-NC antes de usar
```

---

## Checklist Legal Producción

### ✓ Preentrenamiento & Weights

- [x] ViT-Base: Apache-2.0 confirmado (`timm` / `config/teacher_global.yaml`)
- [x] MobileNetV2: Apache-2.0 confirmado (TensorFlow/TF.js)
- [x] ImageNet pesos: Distribuidos con licencia permisiva (usados solo como init)
- [x] Ningún modelo con licencia NC/GPL

### ✓ Fine-tuning Data

- [ ] Dataset filtrado (solo CC0, CC-BY, CC-BY-SA) — actual: 92.4% en `dataset_train.jsonl` / 93,761 muestras estratificadas
- [ ] iNaturalist CC-BY-NC REMOVIDO — pendiente segunda pasada sobre `dataset_train_stratified.jsonl`
- [ ] Perenual no-comercial REMOVIDO — validar por fuente durante filtrado final
- [ ] % de datos permisivos >= 95% — meta aún no alcanzada (faltan ~2.6 pp)

### ✓ Teachers

- [x] Teacher global: Licencia permisiva verificada (`vit_base_patch16_384` Apache-2.0)
- [ ] Teacher regional: Licencia heredada verificada
- [ ] Soft labels: Derivadas solo de teachers permisivos

### ✓ Student

- [ ] Student modelo: Heredado de teachers permisivos ✓
- [ ] Distillation: No introduce nuevas restricciones
- [ ] Fine-tuning: Con data permisiva solo

### ✓ Documentación

- [ ] Manifest lista todas las fuentes + licencias
- [ ] README incluye reconocimientos
- [ ] CC-BY images tienen atribución visible
- [ ] LICENCIAS.md en repo

### ✓ Distribución

- [ ] Modelo puede descargarse libremente (no vendido como SaaS)
- [ ] Atribuciones incluidas en app (pie de página o créditos)
- [ ] No se reclama propiedad intelectual sobre modelos
- [ ] Cambios documentados si se distribuye

---

## Escenario: ¿Qué si usamos Teacher NC para I+D?

### La pregunta

"Quiero usar BioCLIP (NC) para mejorar accuracy en I+D. ¿Contamina el student?"

### La respuesta (conservadora)

**Sí, contamina.** Razones:

1. **Obra derivada:** Soft labels de modelo NC hereda su licencia
2. **Knowledge transfer:** El knowledge del NC se transfiere a student
3. **Legal risk:** Si distribuyés el model, riesgos de demanda

### Solución correcta

**Dos pipelines separados:**

```
EXPERIMENTAL ONLY (I+D interna):
├─ Teachers: cualquier licencia (NC, GPL, etc.)
├─ Student: resultados solo para análisis
└─ NO distribuir, NO monetizar

PRODUCTION (Web pública):
├─ Teachers: Apache-2.0 / MIT / CC0 solo
├─ Student: Seguro para distribuir + monetizar
└─ Documentar abiertamente
```

**En código:**

```python
# config.yaml

environment: "production"  # o "experimental"

if environment == "production":
    ALLOWED_LICENSES = {'Apache-2.0', 'MIT', 'BSD', 'CC0', 'CC-BY'}
    EXPORT_MODEL = True
    MONETIZE_OK = True

elif environment == "experimental":
    ALLOWED_LICENSES = {'*'}  # Cualquiera
    EXPORT_MODEL = False
    MONETIZE_OK = False
    WARNING = "⚠ I+D ONLY - DO NOT DISTRIBUTE"
```

---

## Atribuciones Requeridas

### En README del repo

```markdown
## Model Attribution

This project uses:

**Teachers:**
- ViT-Base: Apache-2.0 (Timm models)
- ImageNet weights: Public domain

**Dataset:**
- iNaturalist: CC-BY, CC-BY-SA (images)
  - See individual attributions in logs
- GBIF: CC0 (public domain)

**Student Model:**
- MobileNetV2 base: Apache-2.0 (TensorFlow)

**Distillation Method:**
- Original research, MIT-licensed

For commercial use, ensure all images are CC0/CC-BY licensed.
```

### En app footer

```html
<footer>
  <p>Plant recognition powered by ML model (Apache-2.0)</p>
  <p>Dataset includes images from:
    <a href="https://www.inaturalist.org">iNaturalist</a> (CC-BY)
    <a href="https://www.gbif.org">GBIF</a> (CC0)
  </p>
</footer>
```

---

## Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|-------------|--------|-----------|
| Usar imagen CC-BY-NC sin filtro | Media | Alto | Automatizar filtro en export |
| Teacher NC contamina student | Baja (si evitas) | Alto | Mantener pipelines separados |
| Atribución CC-BY incompleta | Media | Medio | Agregar step en export: generar manifest |
| Usuario monetiza modelo libremente | Baja | Bajo | Licencia clara en README |

---

## Flujo de Verificación (Automated)

```python
# scripts/license_checker.py

import json
from pathlib import Path

PRODUCTION_LICENSES = {'CC0', 'CC-BY', 'CC-BY-SA', 'Apache-2.0', 'MIT', 'BSD'}

def verify_production_compliance(manifest_file: str):
    """
    Verifica que todo el pipeline es seguro para producción.
    """

    with open(manifest_file) as f:
        manifest = json.load(f)

    issues = []

    # 1. Check teachers
    for teacher in manifest['teachers']:
        if teacher['license'] not in PRODUCTION_LICENSES:
            issues.append(f"Teacher {teacher['name']} has non-permissive license: {teacher['license']}")

    # 2. Check dataset
    non_permissive_images = manifest['dataset']['non_permissive_count']
    total_images = manifest['dataset']['total_count']
    non_permissive_pct = non_permissive_images / total_images

    if non_permissive_pct > 0.05:  # Permite 5%
        issues.append(f"Dataset has {non_permissive_pct*100:.1f}% non-permissive images (limit: 5%)")

    # 3. Check student
    if manifest['student']['inherited_license'] not in PRODUCTION_LICENSES:
        issues.append(f"Student inherited license: {manifest['student']['inherited_license']}")

    # 4. Report
    if issues:
        print("❌ NOT PRODUCTION SAFE:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ PRODUCTION SAFE")
        return True

# Uso:
result = verify_production_compliance('./results/TRAINING_MANIFEST_v1.0.yaml')
if not result:
    sys.exit(1)  # Bloquear deploy
```

---

## Resumen Final

| Aspecto | Acción | Status |
|--------|--------|--------|
| Teachers | Usar solo Apache-2.0 / MIT / CC0 | ✓ |
| Dataset | Filtrar CC-BY-NC, permitir CC0/CC-BY | ✓ |
| Student | Heredará de teachers seguros | ✓ |
| Distillation | No introduce restricciones nuevas | ✓ |
| Documentación | Manifest + README con licencias | ⚠ TO-DO |
| Atribuciones | CC-BY images con atribución visible | ⚠ TO-DO |
| Monetización | Posible con licencias permisivas | ✓ |

**Before deploy to production:**
```bash
python scripts/license_checker.py ./results/TRAINING_MANIFEST_v1.0.yaml
# Must return: ✅ PRODUCTION SAFE
```
