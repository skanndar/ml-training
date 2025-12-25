# Estado de la Conversión a TF.js

**Fecha:** 24 de diciembre de 2025

## Resumen Ejecutivo

✅ **FASES COMPLETADAS:**
- Calibración del modelo (Temperature=2.0, ECE=0.0401)
- Análisis de threshold (recomendado: 0.62 para 94.5% accuracy, 80.1% coverage)
- Cuantización a FP16
- Export a ONNX
- Export a TensorFlow SavedModel
- Validación PyTorch ↔ TensorFlow (MAE=0.0072, max diff=0.045, 100% Top-1 agreement)
- Creación de export_metadata.json con threshold 0.62
- Actualización de documentación (EXPORT_TFJS_PWA.md §1.5)

⏳ **PENDIENTE:**
- Conversión final SavedModel → TF.js (model.json + shards)

## Archivos Disponibles

### Checkpoint Calibrado
```
checkpoints/student_finetune/best_model_temp.pt
  - Temperature: 2.0
  - ECE: 0.0401 (excelente calibración)
  - Listo para inferencia
```

### Modelo Cuantizado FP16
```
results/student_finetune_v1/model_fp16.pt
  - Quantization: float16
  - Tamaño: ~126 MB
```

### Modelos Intermedios (Validados)
```
dist/models/student_v1_fp16_manual/
  ├── student.onnx                    (52.8 MB)
  ├── saved_model/                     (SavedModel validado)
  │   ├── saved_model.pb
  │   ├── variables/
  │   └── assets/
  └── export_metadata.json             (threshold 0.62, calibración)
```

### Validación
```
results/student_finetune_v1/export_validation.json
  - MAE PyTorch vs TensorFlow: 0.0072
  - Max diff: 0.045
  - Top-1 agreement: 100%
```

## Problema Actual: Dependencias Incompatibles

### Opción 1: Python tensorflowjs_converter
**Estado:** ❌ Falla por conflictos de dependencias

TensorFlow 2.13.x (requerido por el converter) tiene dependencias incompatibles con `yggdrasil_decision_forests` y `jax`.

```bash
# Error:
ERROR: Cannot install tensorflow==2.13.1 and ydf==0.13.0 because these
package versions have conflicting dependencies.
```

### Opción 2: Node.js @tensorflow/tfjs-converter
**Estado:** ❌ El paquete no incluye CLI

El paquete `@tensorflow/tfjs-converter` es solo la librería de JavaScript, no incluye el ejecutable `tensorflowjs_converter`.

### Opción 3: Paquete tensorflowjs (Python)
**Estado:** ⏳ Solución recomendada

Requiere crear un venv aislado con versiones específicas. GPT ya generó documentación para esto.

## Solución Recomendada: Google Colab (ACTUALIZADO 24/12/2025)

### ⚠️ Actualización Importante

TensorFlow 2.13.1 **ya no está disponible** en Google Colab. Ahora Colab incluye TensorFlow 2.16+ por defecto.

**Solución:** Usar la versión de TensorFlow ya instalada en Colab (2.17+), que es compatible con `tensorflowjs`.

### Opción 1: Google Colab (RECOMENDADO - 5 minutos)

**✅ Ventajas:**
- Gratis, sin instalación
- TensorFlow ya preinstalado
- Solo instala `tensorflowjs`
- Funciona desde el navegador

**Instrucciones completas:** Ver [GOOGLE_COLAB_CONVERSION.md](GOOGLE_COLAB_CONVERSION.md)

**Pasos rápidos:**

1. **Preparar ZIP localmente:**
   ```bash
   cd /home/skanndar/SynologyDrive/local/aplantida/ml-training
   # Ya creado: saved_model.zip (30 MB)
   ```

2. **Abrir Google Colab:**
   - Ir a https://colab.research.google.com
   - Crear nuevo notebook

3. **Copiar y ejecutar este código en Colab:**
   ```python
   # Instalar tensorflowjs
   !pip install -q tensorflowjs

   # Upload saved_model.zip
   from google.colab import files
   uploaded = files.upload()

   # Descomprimir
   !unzip -q saved_model.zip

   # Convertir con cuantización FP16
   import tensorflowjs as tfjs
   tfjs.converters.convert_tf_saved_model(
       saved_model_dir='./saved_model',
       output_dir='./student_v1_fp16',
       quantization_dtype_map={'float': 'float16'}
   )

   # Comprimir y descargar
   !zip -q -r student_v1_fp16.zip student_v1_fp16
   files.download('student_v1_fp16.zip')
   ```

4. **Descomprimir resultado localmente:**
   ```bash
   unzip ~/Downloads/student_v1_fp16.zip -d dist/models/
   cp dist/models/student_v1_fp16_manual/export_metadata.json \
      dist/models/student_v1_fp16/
   ```

**Tiempo estimado:** 5-8 minutos total

### Opción 2: Conda Environment (Si prefieres local)

⚠️ **Nota:** TensorFlow 2.13.1 puede no estar disponible en repositorios actuales.

```bash
# Crear environment aislado
conda create -n tfjs python=3.10 -y
conda activate tfjs

# Instalar con versiones más recientes
pip install "tensorflow>=2.16" "tensorflowjs"

# Convertir (usar Python API en lugar de CLI)
python << 'EOF'
import tensorflowjs as tfjs
tfjs.converters.convert_tf_saved_model(
    saved_model_dir='dist/models/student_v1_fp16_manual/saved_model',
    output_dir='dist/models/student_v1_fp16',
    quantization_dtype_map={'float': 'float16'}
)
EOF

conda deactivate
```

### Opción 3: Script Python Automatizado

Usar el script preparado:

```bash
# Ver instrucciones en el script
cat scripts/convert_to_tfjs_colab.py

# Copiar contenido completo a una celda de Google Colab y ejecutar
```

## Output Esperado

Una vez completada la conversión, deberías tener:

```
dist/models/student_v1_fp16/
├── model.json                        (Metadata del modelo TF.js)
├── group1-shard1of3.bin              (Pesos, parte 1)
├── group1-shard2of3.bin              (Pesos, parte 2)
├── group1-shard3of3.bin              (Pesos, parte 3)
└── export_metadata.json              (Metadata con threshold 0.62)
```

**Tamaño esperado:** ~50-70 MB total (cuantizado a FP16)

## Verificación Post-Conversión

```javascript
// Test en Node.js
const tf = require('@tensorflow/tfjs-node');

async function test() {
  const model = await tf.loadGraphModel('file://./dist/models/student_v1_fp16/model.json');
  console.log('Model loaded:', model);

  // Test inference
  const dummyInput = tf.randomNormal([1, 224, 224, 3]);
  const output = model.predict(dummyInput);
  console.log('Output shape:', output.shape);  // Debe ser [1, 7120]
}

test();
```

## Siguiente Paso Después de la Conversión

Una vez que tengas `model.json` + shards:

1. **Copiar a frontend:**
   ```bash
   cp -r dist/models/student_v1_fp16 /ruta/a/aplantidaFront/public/models/student_v1.0
   ```

2. **Actualizar PlantRecognition/index.js:**
   ```javascript
   const CONFIDENCE_THRESHOLD = 0.62;
   const MODEL_URLS = {
     'v1.0': '/models/student_v1.0/model.json'
   };
   ```

3. **Actualizar Service Worker:**
   ```javascript
   workbox.precaching.precacheAndRoute([
     {url: '/models/student_v1.0/model.json', revision: 'v1.0'},
     {url: '/models/student_v1.0/group1-shard1of3.bin', revision: 'v1.0'},
     {url: '/models/student_v1.0/group1-shard2of3.bin', revision: 'v1.0'},
     {url: '/models/student_v1.0/group1-shard3of3.bin', revision: 'v1.0'},
   ]);
   ```

4. **Test en navegador:**
   ```javascript
   const model = await tf.loadGraphModel('/models/student_v1.0/model.json');
   ```

## Referencias

- Documentación completa: [EXPORT_TFJS_PWA.md §3.5](aplantida-ml-training/EXPORT_TFJS_PWA.md#35---compatibilidad-del-converter)
- Decisión de threshold: [EXPORT_TFJS_PWA.md §1.5](aplantida-ml-training/EXPORT_TFJS_PWA.md#15---decisión-del-threshold-de-confianza)
- Calibración: [results/student_finetune_v1/temperature_metrics.json](results/student_finetune_v1/temperature_metrics.json)
- Threshold analysis: [results/student_finetune_v1/threshold_analysis_temp.json](results/student_finetune_v1/threshold_analysis_temp.json)

## Comandos de Referencia Rápida

```bash
# Ver tamaño del SavedModel actual
du -sh dist/models/student_v1_fp16_manual/saved_model

# Verificar metadata
cat dist/models/student_v1_fp16_manual/export_metadata.json | jq '.threshold'

# Listar contenido del SavedModel
ls -lah dist/models/student_v1_fp16_manual/saved_model/

# Validación PyTorch vs TensorFlow
cat results/student_finetune_v1/export_validation.json
```

## Estado: LISTO PARA CONVERSIÓN

Todo está preparado. Solo falta ejecutar la conversión SavedModel → TF.js usando una de las opciones A, B o C descritas arriba.

**Progreso:** 90% completado
**Bloqueador:** Dependencias incompatibles en entorno actual (protobuf version conflict: TF 2.15.1 vs TF-DF 1.5.0)
**Solución:** Usar entorno aislado (Docker/Conda/máquina alterna)

## Resumen de lo Completado (24 dic 2025)

### ✅ Configuración Actualizada

- **[config/student.yaml](config/student.yaml):**
  ```yaml
  inference:
    confidence_threshold: 0.62  # Optimized for offline-first PWA
    temperature: 2.0            # Applied during calibration (ECE=0.040)
  ```

- **[dist/models/student_v1_fp16_manual/export_metadata.json](dist/models/student_v1_fp16_manual/export_metadata.json):**
  Metadata completo con:
  - Model version: v1.0
  - Calibration: T=2.0, ECE=0.0401
  - Threshold: 0.62 (accuracy=94.5%, coverage=80.1%)
  - Rationale: "Optimized for offline-first PWA usage in rural areas"
  - Validation metrics: MAE=0.0072, Top-1 agreement=100%

### ✅ Documentación Actualizada

- **[EXPORT_TFJS_PWA.md §1.5](aplantida-ml-training/EXPORT_TFJS_PWA.md#15---decisión-del-threshold-de-confianza):**
  Nueva sección completa explicando:
  - Contexto del threshold
  - Análisis comparativo (0.62 vs 0.66)
  - Justificación técnica (offline-first, calibración excelente)
  - Configuración en código
  - Plan de monitoreo en producción

- **[EXPORT_TFJS_PWA.md §4.1](aplantida-ml-training/EXPORT_TFJS_PWA.md#41---actualizar-plantrecognitionjs):**
  Actualizado `CONFIDENCE_THRESHOLD = 0.62` con comentario explicativo

### ✅ Scripts de Conversión Preparados

- **[scripts/convert_to_tfjs.sh](scripts/convert_to_tfjs.sh):** Intento con Python venv
- **[scripts/convert_to_tfjs_node.sh](scripts/convert_to_tfjs_node.sh):** Intento con Node.js
- **[convert_model.js](convert_model.js):** Script Node.js directo

Todos funcionan conceptualmente pero fallan por dependencias incompatibles en el entorno actual.

### ✅ Archivos Validados y Listos

```
checkpoints/student_finetune/best_model_temp.pt     ← Checkpoint calibrado (T=2.0)
results/student_finetune_v1/model_fp16.pt            ← Cuantizado FP16
dist/models/student_v1_fp16_manual/student.onnx     ← ONNX válido
dist/models/student_v1_fp16_manual/saved_model/     ← SavedModel validado
dist/models/student_v1_fp16_manual/export_metadata.json ← Metadata completo
results/student_finetune_v1/export_validation.json  ← Validación PyTorch↔TF
results/student_finetune_v1/temperature_metrics.json ← Calibración completa
results/student_finetune_v1/threshold_analysis_temp.json ← Análisis threshold
```

## Comando Final (cuando se resuelva entorno)

### Problema Confirmado

Tras múltiples intentos, se confirmó que **tensorflowjs 4.22.0** tiene dependencias circulares insalvables:
- Requiere `tensorflow-decision-forests >= 1.5.0`
- TF-DF 1.5.0 solo es compatible con TensorFlow 2.13.x
- TF 2.13.x tiene conflictos de protobuf con `yggdrasil_decision_forests`
- El entorno actual tiene TF 2.15.1 para el resto del pipeline

**Intentos fallidos:**
1. ❌ Docker con `python:3.10-slim` → falta compilador C
2. ❌ Docker con `python:3.10` → numpy no compila
3. ❌ Docker con `tensorflow/tensorflow:2.13.1` → imagen no existe en Docker Hub
4. ❌ Downgrade protobuf en venv actual → conflicto con ydf
5. ❌ Desinstalar TF-DF → tensorflowjs lo requiere hardcoded
6. ❌ Script Python custom con import bypass → sys.modules hack no funciona

### Solución Recomendada: Máquina Alterna o Conda

**Opción 1: Conda (más confiable)**

```bash
# En otra máquina o en esta con conda instalado
conda create -n tfjs-convert python=3.10 -y
conda activate tfjs-convert

pip install tensorflow==2.13.1 tensorflowjs==4.22.0

tensorflowjs_converter \
  --input_format=tf_saved_model \
  --output_format=tfjs_graph_model \
  --quantize_float16='*' \
  /home/skanndar/SynologyDrive/local/aplantida/ml-training/dist/models/student_v1_fp16_manual/saved_model \
  /home/skanndar/SynologyDrive/local/aplantida/ml-training/dist/models/student_v1_fp16

conda deactivate
```

**Opción 2: Sistema limpio con venv**

Si tienes acceso a otra máquina Ubuntu/Debian sin packages de ML preinstalados:

```bash
# En máquina limpia
cd /tmp
python3 -m venv tfjs
source tfjs/bin/activate

pip install tensorflow==2.13.1 tensorflowjs==4.22.0

# Copiar SavedModel a esta máquina
scp -r user@original:/home/skanndar/.../saved_model ./

# Convertir
tensorflowjs_converter \
  --input_format=tf_saved_model \
  --output_format=tfjs_graph_model \
  --quantize_float16='*' \
  ./saved_model \
  ./output

# Copiar resultado de vuelta
scp -r ./output user@original:/home/skanndar/.../dist/models/student_v1_fp16
```

**Opción 3: Google Colab (gratis, más fácil)**

```python
# Notebook en Google Colab
!pip install tensorflow==2.13.1 tensorflowjs==4.22.0

# Upload saved_model.zip (comprimido previamente)
from google.colab import files
uploaded = files.upload()  # Sube saved_model.zip

!unzip saved_model.zip
!tensorflowjs_converter \
  --input_format=tf_saved_model \
  --output_format=tfjs_graph_model \
  --quantize_float16='*' \
  ./saved_model \
  ./student_v1_fp16

# Download result
!zip -r student_v1_fp16.zip student_v1_fp16
files.download('student_v1_fp16.zip')
```

Esto generará `model.json` + shards listos para deploy en la PWA.

### SavedModel ya está listo

El archivo crítico ya está validado y listo en:
```
dist/models/student_v1_fp16_manual/saved_model/
├── saved_model.pb
├── variables/
│   ├── variables.data-00000-of-00001
│   └── variables.index
└── assets/
```

Solo falta el último paso de conversión a formato TF.js.
