# Export a TF.js y PWA Deployment

## Parte 1: Cuantización

### 1.1 - Opciones de cuantización

| Tipo | Tamaño | Latencia | Accuracy Loss | Soporte |
|------|--------|----------|----------------|---------|
| Float32 | 100% | 1x | 0% | Todos |
| Float16 | 50% | 0.95x | <1% | WebGL/WASM |
| Int8 | 25% | 1.2x (optimizado) | 1-3% | WebGL |

### 1.2 - Recomendación

**FP16:** Balance óptimo para TF.js
- Reduce tamaño a ~50%
- Casi ninguna loss de accuracy (<1%)
- Soportado en navegadores modernos

```bash
# Cuantizar a FP16
python scripts/quantize_model.py \
  --model ./results/student_finetune_v1/best_model.pt \
  --quantization_type fp16 \
  --output ./results/student_finetune_v1/model_fp16.pt

# Resultado: ~40-50MB (vs 80-100MB sin cuantizar)
```

### 1.3 - Int8 (si tamaño es crítico)

```bash
python scripts/quantize_model.py \
  --model ./results/student_finetune_v1/best_model.pt \
  --quantization_type int8 \
  --calibration_data ./data/dataset_splits.jsonl \
  --output ./results/student_finetune_v1/model_int8.pt

# Resultado: ~20-25MB
# PERO: puede perder 2-5% accuracy
# Solo si >100MB es problema

# Validar:
python scripts/compare_quantization.py \
  --original model_fp32.pt \
  --quantized model_int8.pt \
  --test_data test_images.npz
```

### 1.4 - Estado actual (dic 2025)

- Checkpoint FP16 generado con `scripts/quantize_model.py`: `results/student_finetune_v1/model_fp16.pt`.
- `scripts/export_to_tfjs.py` produce `dist/models/student_v1_fp16_manual/student.onnx` + `saved_model/` (incluye `export_metadata.json`).
- Validación PyTorch ↔ TensorFlow guardada en `results/student_finetune_v1/export_validation.json` (MAE 0.007 en 64 muestras).
- **Calibración completada:** Temperature T=2.0, ECE=0.0401, threshold recomendado 0.62 (ver §1.5).

### 1.5 - Decisión del Threshold de Confianza

**Contexto:** El modelo calibrado produce probabilidades confiables (ECE=0.040), pero no todas las predicciones son suficientemente seguras para mostrarlas al usuario. Se implementa un threshold de confianza mínima; predicciones por debajo del threshold retornan "no concluyente" y activan el fallback a PlantNet API.

**Análisis de Opciones:**

| Threshold | Accuracy | Coverage | Implicación |
|-----------|----------|----------|-------------|
| **0.62 (elegido)** | **94.5%** | **80.1%** | De cada 100 predicciones, 80 son aceptadas con 94.5% de exactitud → 76 correctas |
| 0.66 (conservador) | 95.0% | 78.9% | De cada 100 predicciones, 79 son aceptadas con 95.0% de exactitud → 75 correctas |

**Decisión:** **Threshold 0.62**

**Justificación:**

1. **Caso de uso offline-first:** La PWA se usa en zonas rurales sin Internet confiable. Cada "no concluyente" requiere conexión a PlantNet API. Reducir la tasa de "no concluyente" del 21.1% al 19.9% (ganancia de 1.2%) mejora significativamente la UX en modo offline.

2. **Trade-off favorable:** La diferencia de accuracy (0.5%) es imperceptible para usuarios, pero la ganancia de cobertura es tangible. En términos absolutos: con 0.62 obtienes **76 predicciones correctas de 100** vs 75 con 0.66.

3. **Calibración excelente:** Con ECE=0.040, cuando el modelo predice 62% de confianza, realmente es ~62% preciso. El threshold no es arbitrario, está respaldado por métricas de calibración.

4. **Ajustable en producción:** El threshold se puede ajustar dinámicamente según feedback de usuarios. Si la tasa de falsos positivos es inaceptable, se puede subir a 0.66 con una actualización del Service Worker.

**Configuración en código:**

```javascript
// aplantidaFront/src/components/PlantRecognition/index.js
const CONFIDENCE_THRESHOLD = 0.62;

// En export_metadata.json
"threshold": {
  "value": 0.62,
  "accuracy": 0.945,
  "coverage": 0.801,
  "rationale": "Optimized for offline-first PWA usage in rural areas"
}
```

**Plan de monitoreo:**

- Registrar todas las predicciones con: `{confidence, predicted_species, user_feedback}`
- Revisar mensualmente:
  - Si accuracy real < 94% → subir threshold a 0.66
  - Si tasa de "no concluyente" > 25% → bajar threshold a 0.60
  - Si usuarios en zonas urbanas reportan baja precisión → implementar A/B testing con ambos thresholds

**Referencia:** Ver análisis completo en `results/student_finetune_v1/threshold_analysis_temp.json`

---

## Parte 2: Export a SavedModel (TensorFlow format)

> Usa `scripts/export_to_tfjs.py` para automatizar PyTorch → ONNX → SavedModel antes de convertir a TF.js.

```bash
./venv/bin/python scripts/export_to_tfjs.py \
  --config config/student.yaml \
  --model results/student_finetune_v1/model_fp16.pt \
  --class-mapping ./data/class_mapping.json \
  --output-dir ./dist/models/student_v1_fp16_manual \
  --quantization float16 \
  --force
```

Genera: `student.onnx`, `saved_model/`, `export_metadata.json` y, si las dependencias son compatibles, ejecuta `tensorflowjs_converter` automáticamente.

### 2.1 - PyTorch → SavedModel

```python
# scripts/export_to_savedmodel.py

import torch
import tensorflow as tf
import torch.onnx
import onnx
import onnx_tf.backend

def pytorch_to_savedmodel(
    pytorch_model_path: str,
    output_dir: str,
    input_shape: tuple = (1, 3, 224, 224)
):
    """
    Convierte PyTorch → ONNX → TensorFlow SavedModel
    """

    # 1. Cargar modelo PyTorch
    model = torch.jit.load(pytorch_model_path)
    model.eval()

    # 2. Crear dummy input
    dummy_input = torch.randn(input_shape)

    # 3. Exportar a ONNX
    onnx_path = f"{output_dir}/model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=13,
        input_names=['input_image'],
        output_names=['output_logits'],
        dynamic_axes={
            'input_image': {0: 'batch_size'},
            'output_logits': {0: 'batch_size'}
        }
    )

    # 4. Validar ONNX
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model valid: {onnx_path}")

    # 5. Convertir a TensorFlow
    onnx_tf_rep = onnx_tf.backend.prepare(onnx_model)
    onnx_tf_rep.export_graph(f"{output_dir}/saved_model")

    print(f"SavedModel exported to {output_dir}/saved_model")
```

### 2.2 - Alternativa directa: TorchScript a TF.js

```python
# Más directo: PyTorch → TF.js converter

import subprocess

def pytorch_to_tfjs(
    pytorch_path: str,
    output_dir: str
):
    """
    Usa tfjs converter (requiere instalar tfjs-converter)
    """

    # Instalar si no existe
    subprocess.run(
        ["pip", "install", "tensorflowjs"],
        check=True
    )

    # Convertir SavedModel → TF.js
    subprocess.run([
        "tensorflowjs_converter",
        "--input_format", "tf_saved_model",
        "--output_format", "tfjs_graph_model",
        f"{pytorch_path}/saved_model",
        output_dir
    ], check=True)

    print(f"TF.js model exported to {output_dir}")
```

---

## Parte 3: Conversión a TF.js

### 3.0 - Scripts disponibles

- `scripts/export_to_tfjs.py`: corre PyTorch → ONNX → SavedModel + (opcional) converter.
- `scripts/validate_tfjs_export.py`: asegura que el SavedModel mantiene las probabilidades vs PyTorch antes de subirlo a TF.js.
- Si el converter falla por dependencias, crea un venv exclusivo para TF 2.13.x o usa el CLI de Node (`npx @tensorflow/tfjs-converter`).

### 3.1 - Comandos de conversión

```bash
# Opción 1: SavedModel → TF.js
tensorflowjs_converter \
  --input_format tf_saved_model \
  --output_format tfjs_graph_model \
  ./results/student_fp16/saved_model \
  ./dist/models/student_v1.0

# Opción 2: Keras H5 → TF.js
tensorflowjs_converter \
  --input_format keras \
  ./results/student_fp16/model.h5 \
  ./dist/models/student_v1.0

# Output esperado:
# dist/models/student_v1.0/
# ├─ model.json           (metadatos + arquitectura)
# ├─ group1-shard1of3.bin (pesos, shard 1)
# ├─ group1-shard2of3.bin (pesos, shard 2)
# └─ group1-shard3of3.bin (pesos, shard 3)
```

### 3.2 - Validar equivalencia

```python
# scripts/validate_tfjs_export.py

import numpy as np
import torch
import json

def validate_export(
    pytorch_model_path: str,
    tfjs_model_dir: str,
    test_images_path: str
):
    """
    Compara predicciones Python vs TF.js
    """

    # 1. Cargar imágenes de test
    test_data = np.load(test_images_path)
    images = test_data['images']  # (100, 224, 224, 3)

    # 2. Predicciones Python
    model = torch.load(pytorch_model_path)
    model.eval()

    python_predictions = []
    with torch.no_grad():
        for img in images:
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
            logits = model(tensor).numpy()
            python_predictions.append(logits)

    python_predictions = np.concatenate(python_predictions, axis=0)  # (100, 9000)

    # 3. Predicciones TF.js (via Node.js script)
    # Crear test_tfjs.js
    with open("test_tfjs.js", "w") as f:
        f.write(f"""
const tf = require('@tensorflow/tfjs');
const fs = require('fs');

async function testModel() {{
    const model = await tf.loadLayersModel('file://{tfjs_model_dir}/model.json');

    const testData = {{}};  // Cargar imágenes aquí

    const predictions = [];
    for (let i = 0; i < testData.length; i++) {{
        const logits = model.predict(tf.tensor(testData[i]));
        predictions.push(await logits.data());
    }}

    fs.writeFileSync('tfjs_predictions.json', JSON.stringify(predictions));
}}

testModel();
""")

    subprocess.run(["node", "test_tfjs.js"], check=True)

    # 4. Cargar predicciones TF.js
    with open("tfjs_predictions.json") as f:
        tfjs_predictions = np.array(json.load(f))

    # 5. Comparar
    correlation = np.corrcoef(
        python_predictions.flatten(),
        tfjs_predictions.flatten()
    )[0, 1]

    mae = np.mean(np.abs(python_predictions - tfjs_predictions))
    mse = np.mean((python_predictions - tfjs_predictions) ** 2)

    print(f"Correlation: {correlation:.6f}")  # Debe ser > 0.99
    print(f"MAE: {mae:.6f}")                  # Debe ser < 0.01
    print(f"MSE: {mse:.8f}")

    if correlation > 0.99:
        print("✓ Export válido!")
    else:
        print("✗ Revisar export (correlation too low)")
```

### 3.5 - Compatibilidad del converter

- `tensorflowjs_converter` (Python) requiere TensorFlow 2.13.x + `tensorflow_decision_forests 1.5.0` + `yggdrasil_decision_forests`. Con TF 2.15+ aparecen errores (`ModuleNotFoundError: yggdrasil_decision_forests`, `jax` incompatibles, etc.).
- Recomendación: crea un venv limpio y ejecuta:
  ```bash
  python -m venv tfjs-env && source tfjs-env/bin/activate
  pip install "tensorflow==2.13.1" tensorflowjs==4.22.0 \
              tensorflow-decision-forests==1.5.0 ydf==0.13.0 \
              tensorflow-hub==0.16.1 tensorflow-addons==0.21.0
  tensorflowjs_converter --quantize_float16 '*' \
    dist/models/student_v1_fp16_manual/saved_model \
    dist/models/student_v1_fp16
  ```
- Alternativa: instalar `@tensorflow/tfjs-converter` global/Node (`npm install -g @tensorflow/tfjs-converter`) y ejecutar `tfjs-converter`/`npx` apuntando al SavedModel generado en Python.

---

## Parte 4: Integración con PWA

### 4.1 - Actualizar PlantRecognition.js

```javascript
// aplantidaFront/src/components/PlantRecognition/index.js

import React, { useState, useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';

// URLs del modelo deployado (CDN o local)
const MODEL_URLS = {
  'v1.0': '/models/student_v1.0/model.json',
  'v1.1': '/models/student_v1.1/model.json'
};

const CURRENT_MODEL_VERSION = 'v1.0';
const CONFIDENCE_THRESHOLD = 0.62;  // Calibrated threshold (94.5% accuracy, 80.1% coverage)

// Mapping de class ID → plant info (cargado desde backend)
let classLabelMap = null;

async function loadClassLabelMap() {
  const response = await fetch('/api/plants/class-mapping');
  classLabelMap = await response.json();  // {0: {id, latinName, commonName}, ...}
}

async function loadModel(version = CURRENT_MODEL_VERSION) {
  try {
    // Intenta cargar desde IndexedDB (caché persistente)
    const modelUrl = `indexeddb://aplantida-student-${version}`;

    try {
      return await tf.loadLayersModel(modelUrl);
    } catch (e) {
      console.log(`Model not in IndexedDB, downloading from CDN...`);

      // Descargar desde CDN
      const model = await tf.loadLayersModel(MODEL_URLS[version]);

      // Guardar en IndexedDB para futuro
      await model.save(modelUrl);

      return model;
    }
  } catch (error) {
    console.error(`Failed to load model v${version}:`, error);
    throw error;
  }
}

async function recognizeWithStudent(imageElement) {
  try {
    const model = await loadModel(CURRENT_MODEL_VERSION);

    // Preprocesar imagen
    let tensor = tf.browser.fromPixels(imageElement);

    // Resize a 224x224
    tensor = tf.image.resizeBilinear(tensor, [224, 224]);

    // Normalizar a [0, 1]
    tensor = tensor.div(255.0);

    // Añadir batch dimension
    tensor = tensor.expandDims(0);

    // Inferencia
    const logits = model.predict(tensor);

    // Softmax + probabilidades
    const probs = tf.softmax(logits);

    // Top-5
    const topK = tf.topk(probs, 5);
    const indices = topK.indices.dataSync();
    const values = topK.values.dataSync();

    // Mapear a plant IDs
    const results = [];
    for (let i = 0; i < 5; i++) {
      const classId = indices[i];
      const confidence = values[i];

      if (confidence < CONFIDENCE_THRESHOLD && i === 0) {
        // No conclusivo
        return {
          success: false,
          message: `No conclusive (confidence ${confidence.toFixed(2)} < ${CONFIDENCE_THRESHOLD})`,
          results: null
        };
      }

      const plantInfo = classLabelMap[classId];
      results.push({
        plant_id: plantInfo.id,
        latin_name: plantInfo.latinName,
        common_name: plantInfo.commonName,
        confidence: confidence,
        rank: i + 1
      });
    }

    // Limpiar
    tensor.dispose();
    logits.dispose();
    probs.dispose();
    topK.dispose();

    return {
      success: true,
      results: results,
      model_version: CURRENT_MODEL_VERSION
    };

  } catch (error) {
    console.error('Student recognition error:', error);
    throw error;
  }
}

export default function PlantRecognition() {
  const [loading, setLoading] = useState(false);
  const [modelLoading, setModelLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [modelVersion, setModelVersion] = useState(CURRENT_MODEL_VERSION);

  // Precargar modelo al montar
  useEffect(() => {
    const preload = async () => {
      try {
        setModelLoading(true);
        await loadClassLabelMap();
        await loadModel(CURRENT_MODEL_VERSION);
        setModelLoading(false);
      } catch (err) {
        console.error('Failed to preload:', err);
        setModelLoading(false);
      }
    };

    preload();
  }, []);

  const handleRecognize = async (imageFile, captureMethod) => {
    try {
      setLoading(true);

      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.src = URL.createObjectURL(imageFile);

      img.onload = async () => {
        try {
          // Primero intentar con Student (TF.js)
          let recognitionResults = await recognizeWithStudent(img);

          if (!recognitionResults.success) {
            // Si no conclusivo, usar API PlantNet como fallback
            const apiResult = await axiosRequestFunctions.recognizePlant(
              imageFile,
              captureMethod
            );
            recognitionResults = apiResult.recognition.results || [];
          }

          setResults(recognitionResults);
          setLoading(false);
        } catch (err) {
          console.error('Recognition error:', err);
          setLoading(false);
        }
      };
    } catch (err) {
      console.error('Error in handleRecognize:', err);
      setLoading(false);
    }
  };

  return (
    <div>
      {modelLoading && (
        <Alert
          message="Loading ML Model"
          description={`Loading student model v${CURRENT_MODEL_VERSION}...`}
          type="info"
        />
      )}

      {/* Resto del componente */}
    </div>
  );
}
```

### 4.2 - Service Worker para caching

```javascript
// public/service-worker.js (Workbox)

importScripts('https://storage.googleapis.com/workbox-cdn/releases/6.5.4/workbox-sw.js');

// Precaché de assets + modelo
workbox.precaching.precacheAndRoute([
  {url: '/models/student_v1.0/model.json', revision: 'v1.0'},
  {url: '/models/student_v1.0/group1-shard1of3.bin', revision: 'v1.0'},
  {url: '/models/student_v1.0/group1-shard2of3.bin', revision: 'v1.0'},
  {url: '/models/student_v1.0/group1-shard3of3.bin', revision: 'v1.0'},
  // ... resto de assets
]);

// Cache runtime para model updates
workbox.routing.registerRoute(
  ({request}) => request.destination === 'image' && request.url.includes('/models/'),
  new workbox.strategies.CacheFirst({
    cacheName: 'tf-models',
    plugins: [
      new workbox.expiration.ExpirationPlugin({
        maxAgeSeconds: 30 * 24 * 60 * 60  // 30 días
      })
    ]
  })
);

// Message handler: actualización de modelo disponible
self.addEventListener('message', event => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});
```

### 4.3 - Model Update Mechanism

```javascript
// Detectar nueva versión disponible

// En App.js o principal
async function checkModelUpdates() {
  try {
    const response = await fetch('/api/models/latest-version');
    const {version} = await response.json();

    if (version !== CURRENT_MODEL_VERSION) {
      console.log(`New model available: v${version}`);

      // Notificar usuario
      showUpdateNotification({
        title: 'ML Model Update Available',
        message: `A new version (v${version}) is available. Update to improve accuracy!`,
        onAccept: async () => {
          // Descargar nuevo modelo en background
          await preloadModel(version);

          // Recargar app
          location.reload();
        }
      });
    }
  } catch (err) {
    console.warn('Could not check for model updates:', err);
  }
}

// Ejecutar check cada 24h o al cargar
setInterval(checkModelUpdates, 24 * 60 * 60 * 1000);
checkModelUpdates();  // On load
```

---

## Parte 5: Backend API

### 5.1 - Endpoint para mapeo clase ↔ planta

```javascript
// aplantidaBack/routes/plants-router.js

router.get('/class-mapping', async (req, res) => {
  try {
    const plants = await Plant.find({})
      .select('_id latinName commonName')
      .sort('_id');

    // Mapeo: índice → plant
    const mapping = {};
    plants.forEach((plant, index) => {
      mapping[index] = {
        id: plant._id.toString(),
        latinName: plant.latinName,
        commonName: plant.commonName
      };
    });

    // Cachear en browser por 30 días
    res.set('Cache-Control', 'public, max-age=2592000');
    res.json(mapping);
  } catch (err) {
    res.status(500).json({error: err.message});
  }
});

// Endpoint para nuevo modelo
router.get('/models/latest-version', (req, res) => {
  res.json({
    version: 'v1.0',  // Lee de config o BD
    releaseDate: '2024-12-17',
    releaseNotes: 'Improved regional accuracy for Spain'
  });
});
```

---

## Parte 6: Despliegue en Producción

### 6.1 - Estructura de carpetas

```
aplantidaFront/
├── public/
│   ├── models/
│   │   ├── student_v1.0/
│   │   │   ├── model.json
│   │   │   ├── group1-shard1of3.bin
│   │   │   ├── group1-shard2of3.bin
│   │   │   └── group1-shard3of3.bin
│   │   ├── student_v1.1/  (futuras versiones)
│   │   └── manifest.json
│   └── service-worker.js
├── src/
│   └── components/
│       └── PlantRecognition/
│           └── index.js
```

### 6.2 - Deploy checklist

- [ ] Modelo export validado (correlation > 0.99)
- [ ] Tamaño total <= 150MB (cuantizado)
- [ ] Service Worker registrado
- [ ] IndexedDB caché configurado
- [ ] Class mapping endpoint funciona
- [ ] Model update checks enabled
- [ ] Fallback a PlantNet API si falla
- [ ] Tests en múltiples navegadores
- [ ] Tests offline mode funciona
- [ ] Métricas de latencia < 2s GPU, < 5s CPU

---

## Troubleshooting

**Error: "Model not found" en browser**
→ Verificar CORS en CDN
→ Verificar ruta exacta en model.json

**Inference lento (> 5s)**
→ Usar WebGL backend (no CPU)
→ Verificar tamaño modelo (puede no estar cuantizado)

**Modelo no cachea en IndexedDB**
→ Verificar quota: Chrome 50MB por origen, Firefox 1GB
→ Si modelo > 50MB, necesita chunked upload

**Diferentes predicciones Python vs TF.js**
→ Verificar normalización de imágenes idéntica
→ Verificar input shape (224, 224, 3)
→ Comparar con `validate_tfjs_export.py`
