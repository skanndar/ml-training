# Cómo Convertir a TF.js con Google Colab

## 🚀 Inicio Rápido (5 minutos)

### Paso 1: Abrir Google Colab

Ir a: **https://colab.research.google.com**

### Paso 2: Crear Nuevo Notebook

Click en: **Archivo → Nuevo cuaderno**

### Paso 3: Copiar y Pegar este Código

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

### Paso 4: Ejecutar

1. Click en el botón **▶ Ejecutar** (o presionar `Ctrl+Enter`)
2. Cuando pida "Choose Files", seleccionar: **`saved_model.zip`** (ubicado en este directorio)
3. Esperar 3-5 minutos
4. Descargar `student_v1_fp16.zip` automáticamente

### Paso 5: Descomprimir Resultado

```bash
cd /home/skanndar/SynologyDrive/local/aplantida/ml-training
unzip ~/Downloads/student_v1_fp16.zip -d dist/models/
cp dist/models/student_v1_fp16_manual/export_metadata.json dist/models/student_v1_fp16/
```

## ✅ Verificar

```bash
ls -lh dist/models/student_v1_fp16/
```

Deberías ver:
- `model.json` (~5 KB)
- `group1-shard*.bin` (varios archivos, ~50 MB total)
- `export_metadata.json` (~1 KB)

---

## 📚 Documentación Completa

- **Guía detallada:** [GOOGLE_COLAB_CONVERSION.md](GOOGLE_COLAB_CONVERSION.md)
- **Script automatizado:** [scripts/convert_to_tfjs_colab.py](scripts/convert_to_tfjs_colab.py)
- **Resumen actualizado:** [RESUMEN_CONVERSION_ACTUALIZADO.md](RESUMEN_CONVERSION_ACTUALIZADO.md)
- **Estado general:** [CONVERSION_STATUS.md](CONVERSION_STATUS.md)

## ❓ Problemas Comunes

**Error: "No matching distribution found for tensorflow==2.13.1"**
- Solución: Usar el código arriba (usa TF 2.17+ ya instalado en Colab)

**Error al descargar**
- Solución: Click derecho en `student_v1_fp16.zip` en panel izquierdo → Descargar

**Archivo muy pequeño**
- Solución: Revisar logs de la celda, buscar errores en la conversión

---

**Tiempo total:** 5-8 minutos
**Archivos necesarios:** `saved_model.zip` (30 MB, ya creado en este directorio)
