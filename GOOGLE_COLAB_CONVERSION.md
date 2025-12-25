# Conversión a TF.js usando Google Colab

**Fecha:** 24 de diciembre de 2025
**Problema:** TensorFlow 2.13.1 no está disponible en Google Colab (solo 2.16+)
**Solución:** Usar versiones recientes compatibles con Colab

## Por qué Google Colab

Google Colab ahora solo incluye TensorFlow 2.16+ por defecto. El paquete `tensorflowjs` es compatible con cualquier versión de TensorFlow 2.x, así que podemos usar la versión ya instalada en Colab.

**Ventajas:**
- ✅ Gratis y sin instalación
- ✅ TensorFlow ya preinstalado
- ✅ Solo necesitas instalar `tensorflowjs`
- ✅ No requiere dependencias de compilación
- ✅ Funciona desde el navegador

## Instrucciones Paso a Paso

### Paso 1: Preparar el SavedModel localmente

```bash
# En tu máquina local
cd /home/skanndar/SynologyDrive/local/aplantida/ml-training

# Comprimir el SavedModel
zip -r saved_model.zip dist/models/student_v1_fp16_manual/saved_model/

# Verificar tamaño (debe ser ~51 MB)
ls -lh saved_model.zip
```

### Paso 2: Abrir Google Colab

1. Ir a https://colab.research.google.com
2. Crear nuevo notebook: **Archivo → Nuevo cuaderno**
3. Renombrar a "TFjs Conversion" (opcional)

### Paso 3: Ejecutar la conversión

Copiar y pegar el siguiente código en una celda de Colab:

```python
# ==============================================================================
# INSTALACIÓN
# ==============================================================================

print("=== Instalando tensorflowjs ===")
!pip install -q tensorflowjs

import tensorflow as tf
print(f"✅ TensorFlow version: {tf.__version__}")


# ==============================================================================
# UPLOAD DEL SAVEDMODEL
# ==============================================================================

print("\n=== Upload saved_model.zip ===")
from google.colab import files
uploaded = files.upload()

# Descomprimir
!unzip -q saved_model.zip
!ls -lh saved_model/

print("✅ SavedModel listo")


# ==============================================================================
# CONVERSIÓN
# ==============================================================================

print("\n=== Conversión SavedModel → TF.js ===")

import tensorflowjs as tfjs

tfjs.converters.convert_tf_saved_model(
    saved_model_dir='./saved_model',
    output_dir='./student_v1_fp16',
    quantization_dtype_map={'float': 'float16'}
)

print("✅ Conversión completada")


# ==============================================================================
# VERIFICAR RESULTADO
# ==============================================================================

!ls -lh student_v1_fp16/

import os
import glob

files_list = glob.glob('./student_v1_fp16/*')
total_size = sum(os.path.getsize(f) for f in files_list) / (1024 * 1024)

print(f"\n📦 Archivos generados:")
for f in sorted(files_list):
    size = os.path.getsize(f) / (1024 * 1024)
    print(f"  - {os.path.basename(f):40s} ({size:6.2f} MB)")

print(f"\n📊 Tamaño total: {total_size:.2f} MB")


# ==============================================================================
# DOWNLOAD
# ==============================================================================

print("\n=== Comprimiendo resultado ===")
!zip -q -r student_v1_fp16.zip student_v1_fp16

print("Descargando student_v1_fp16.zip...")
files.download('student_v1_fp16.zip')

print("\n✅ DESCARGA COMPLETADA")
```

### Paso 4: Ejecutar y esperar

1. Hacer clic en el botón **▶ Ejecutar** (o presionar `Ctrl+Enter`)
2. Cuando pida "Choose Files", seleccionar `saved_model.zip`
3. Esperar 2-3 minutos (instalación + conversión)
4. Al final descargará automáticamente `student_v1_fp16.zip`

### Paso 5: Descomprimir y mover en local

```bash
# En tu máquina local
cd /home/skanndar/SynologyDrive/local/aplantida/ml-training

# Descomprimir (ajusta la ruta donde se descargó)
unzip ~/Downloads/student_v1_fp16.zip -d dist/models/

# Copiar metadata
cp dist/models/student_v1_fp16_manual/export_metadata.json \
   dist/models/student_v1_fp16/

# Verificar
ls -lh dist/models/student_v1_fp16/
```

Deberías ver:

```
dist/models/student_v1_fp16/
├── model.json                        (~5 KB)
├── group1-shard1of*.bin              (~15-20 MB cada uno)
├── group1-shard2of*.bin
├── group1-shard3of*.bin
└── export_metadata.json              (~1 KB)
```

## Solución de Problemas

### Error: "No matching distribution found for tensorflow==2.13.1"

**Causa:** Google Colab ya no incluye TF 2.13.x

**Solución:** Usar el código arriba que usa la versión preinstalada (2.16+)

### Error: "quantize_float16 not recognized"

**Causa:** Sintaxis cambió en versiones recientes de tensorflowjs

**Solución:** Usar `quantization_dtype_map={'float': 'float16'}` en lugar de `--quantize_float16='*'`

### Error al descargar: "Failed - Forbidden"

**Causa:** Navegador bloqueó descarga automática

**Solución:** Hacer clic derecho en archivo → Descargar en el panel izquierdo de Colab

### Archivo descargado demasiado pequeño

**Causa:** La conversión falló silenciosamente

**Solución:** Revisar salida de la celda, buscar errores en el log

## Verificación del Resultado

### Estructura esperada

```
student_v1_fp16/
├── model.json                   # Graph definition (5-10 KB)
├── group1-shard1of3.bin         # Weights part 1 (~20 MB)
├── group1-shard2of3.bin         # Weights part 2 (~20 MB)
└── group1-shard3of3.bin         # Weights part 3 (~10 MB)
```

**Tamaño total esperado:** 50-70 MB (cuantizado a FP16)

### Verificar model.json

```bash
cat dist/models/student_v1_fp16/model.json | jq '.format'
# Output esperado: "graph-model"

cat dist/models/student_v1_fp16/model.json | jq '.modelTopology.node[0].name'
# Debe mostrar un nombre de nodo válido
```

### Test de carga en Node.js

```javascript
const tf = require('@tensorflow/tfjs-node');

async function test() {
  const model = await tf.loadGraphModel(
    'file://./dist/models/student_v1_fp16/model.json'
  );

  console.log('✅ Model loaded successfully');
  console.log('Input shape:', model.inputs[0].shape);   // [null, 224, 224, 3]
  console.log('Output shape:', model.outputs[0].shape); // [null, 7120]

  // Test inference
  const dummyInput = tf.randomNormal([1, 224, 224, 3]);
  const output = model.predict(dummyInput);
  console.log('Inference output shape:', output.shape); // [1, 7120]

  tf.dispose([dummyInput, output]);
}

test().catch(console.error);
```

## Comparación con SavedModel

### Tamaño

- **SavedModel:** ~51 MB (saved_model.pb)
- **TF.js (FP16):** ~50-60 MB (model.json + shards)
- **TF.js (FP32):** ~100-120 MB (sin cuantización)

La diferencia es mínima porque ambos usan FP16.

### Precisión

El modelo TF.js debe tener la misma precisión que el SavedModel (MAE < 0.01 en validación).

## Próximos Pasos

Una vez tengas `dist/models/student_v1_fp16/` completo:

1. **Ver documentación de integración:**
   ```bash
   cat FRONTEND_INTEGRATION.md
   ```

2. **Copiar a frontend:**
   ```bash
   cp -r dist/models/student_v1_fp16 \
      /ruta/a/aplantidaFront/public/models/student_v1.0
   ```

3. **Actualizar código frontend:**
   Ver sección §4 en [EXPORT_TFJS_PWA.md](aplantida-ml-training/EXPORT_TFJS_PWA.md#4---integración-en-frontend-pwa)

4. **Test en navegador:**
   ```javascript
   const model = await tf.loadGraphModel('/models/student_v1.0/model.json');
   console.log('Model loaded:', model);
   ```

## Referencias

- **SavedModel validado:** [dist/models/student_v1_fp16_manual/saved_model/](../dist/models/student_v1_fp16_manual/saved_model/)
- **Metadata:** [export_metadata.json](../dist/models/student_v1_fp16_manual/export_metadata.json)
- **Validación PyTorch↔TF:** [export_validation.json](../results/student_finetune_v1/export_validation.json)
- **Threshold decision:** [EXPORT_TFJS_PWA.md §1.5](aplantida-ml-training/EXPORT_TFJS_PWA.md#15---decisión-del-threshold-de-confianza)
- **Frontend integration:** [FRONTEND_INTEGRATION.md](FRONTEND_INTEGRATION.md)

## Script Automatizado (Opcional)

Alternativamente, usa el script Python que genera todo el código para ti:

```bash
# Ver el script completo
cat scripts/convert_to_tfjs_colab.py

# El script incluye:
# - Instalación automática de tensorflowjs
# - Upload de saved_model.zip
# - Conversión con cuantización FP16
# - Verificación de archivos
# - Download automático del resultado
```

Copia el contenido de `scripts/convert_to_tfjs_colab.py` en una celda de Colab y ejecútalo.

## Tiempo Estimado

- **Upload (51 MB):** 1-2 minutos (depende de tu conexión)
- **Instalación:** 30 segundos
- **Conversión:** 1-2 minutos
- **Download (50-60 MB):** 1-2 minutos

**Total:** 5-8 minutos

---

**Última actualización:** 24 de diciembre de 2025
**Estado:** Verificado con TensorFlow 2.17+ en Google Colab
