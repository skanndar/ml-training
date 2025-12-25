# Integración del Modelo Student en Frontend

**Estado:** Listo para deploy una vez que tengas `model.json` + shards del TF.js converter

---

## Paso 1: Copiar Modelo a Frontend

Una vez que hayas ejecutado el `tensorflowjs_converter` (ver `CONVERSION_STATUS.md` opciones Conda/Colab/máquina alterna) y tengas los archivos:

```
dist/models/student_v1_fp16/
├── model.json
├── group1-shard1of*.bin
├── group1-shard2of*.bin
└── ...
```

Cópialos al frontend:

```bash
cd /home/skanndar/SynologyDrive/local/aplantida/ml-training

# Crear directorio models en frontend
mkdir -p ../aplantidaFront/public/models/student_v1.0

# Copiar modelo convertido
cp -r dist/models/student_v1_fp16/* ../aplantidaFront/public/models/student_v1.0/

# Copiar también el metadata
cp dist/models/student_v1_fp16_manual/export_metadata.json ../aplantidaFront/public/models/student_v1.0/

# Verificar
ls -lh ../aplantidaFront/public/models/student_v1.0/
```

---

## Paso 2: Actualizar PlantRecognition Component

**Archivo:** `/home/skanndar/SynologyDrive/local/aplantida/aplantidaFront/src/components/PlantRecognition/index.js`

### Cambios necesarios:

#### 2.1 - Imports y constantes

```javascript
import React, { useState, useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
// REMOVED: import * as mobilenet from '@tensorflow-models/mobilenet';
import { Tabs, Spin, message, Card, Alert } from 'antd';
import AplantidaIcon from '../AplantidaIcon';
import CameraCapture from './CameraCapture';
import ImageUpload from './ImageUpload';
import RecognitionResults from './RecognitionResults';
import RecognitionHistory from './RecognitionHistory';
import axiosRequestFunctions from '../../lib/auth-service';

// UPDATED: Usar modelo calibrado con threshold 0.62
const CONFIDENCE_THRESHOLD = 0.62;  // Calibrated (94.5% accuracy, 80.1% coverage)

// Model URLs
const MODEL_URLS = {
  'v1.0': '/models/student_v1.0/model.json',
};
const CURRENT_MODEL_VERSION = 'v1.0';

// Class mapping (loaded from backend)
let classLabelMap = null;

// Model loading state
let model = null;
let modelLoading = false;
```

#### 2.2 - Cargar class mapping desde backend

```javascript
async function loadClassLabelMap() {
  if (classLabelMap) return classLabelMap;

  try {
    const response = await axiosRequestFunctions.getPlantClassMapping();
    classLabelMap = response.data;  // {0: {id, latinName, commonName}, ...}
    console.log(`✓ Loaded ${Object.keys(classLabelMap).length} plant classes`);
    return classLabelMap;
  } catch (error) {
    console.error('Failed to load class mapping:', error);
    throw error;
  }
}
```

#### 2.3 - Cargar student model

```javascript
async function loadModel() {
  if (model) return model;
  if (modelLoading) return new Promise((resolve) => {
    const checkModel = setInterval(() => {
      if (model) {
        clearInterval(checkModel);
        resolve(model);
      }
    }, 100);
  });

  modelLoading = true;
  try {
    // Set backend preference
    const backends = ['webgl', 'cpu'];
    for (const backend of backends) {
      try {
        await tf.setBackend(backend);
        console.log(`✓ TensorFlow.js backend set to: ${backend}`);
        break;
      } catch (err) {
        console.warn(`Backend ${backend} not available, trying next...`);
        if (backend === backends[backends.length - 1]) {
          throw new Error('No suitable TensorFlow backend found');
        }
      }
    }

    // Try to load from IndexedDB cache first
    const modelUrl = `indexeddb://aplantida-student-${CURRENT_MODEL_VERSION}`;

    try {
      model = await tf.loadGraphModel(modelUrl);
      console.log(`✓ Student model v${CURRENT_MODEL_VERSION} loaded from cache`);
    } catch (e) {
      console.log('Model not in cache, downloading from server...');

      // Download from server
      const serverUrl = MODEL_URLS[CURRENT_MODEL_VERSION];
      model = await tf.loadGraphModel(serverUrl);

      // Save to IndexedDB for future use
      try {
        await model.save(modelUrl);
        console.log('✓ Model cached in IndexedDB');
      } catch (cacheErr) {
        console.warn('Failed to cache model:', cacheErr);
      }
    }

    modelLoading = false;
    console.log('✓ Student model loaded successfully');
    return model;
  } catch (error) {
    modelLoading = false;
    console.error('Error loading model:', error);
    throw error;
  }
}
```

#### 2.4 - Reconocimiento con student model

```javascript
async function recognizeWithStudent(imageElement) {
  try {
    // Ensure class mapping is loaded
    const mapping = await loadClassLabelMap();

    // Load model
    const loadedModel = await loadModel();

    // Preprocess image
    let tensor = tf.browser.fromPixels(imageElement);

    // Resize to 224x224 (student model input size)
    tensor = tf.image.resizeBilinear(tensor, [224, 224]);

    // Normalize to [0, 1]
    tensor = tensor.div(255.0);

    // Add batch dimension
    tensor = tensor.expandDims(0);

    // Run inference
    const logits = loadedModel.predict(tensor);

    // Apply softmax to get probabilities
    const probs = tf.softmax(logits);

    // Get top-5 predictions
    const { values, indices } = tf.topk(probs, 5);

    const probsArray = await values.data();
    const indicesArray = await indices.data();

    // Clean up tensors
    tensor.dispose();
    logits.dispose();
    probs.dispose();
    values.dispose();
    indices.dispose();

    // Map to plant info
    const results = [];
    for (let i = 0; i < 5; i++) {
      const classIdx = indicesArray[i];
      const confidence = probsArray[i];

      // Check threshold on best prediction
      if (i === 0 && confidence < CONFIDENCE_THRESHOLD) {
        return {
          success: false,
          message: `No conclusive (confidence ${(confidence * 100).toFixed(1)}% < ${(CONFIDENCE_THRESHOLD * 100).toFixed(0)}%)`,
          confidence: confidence,
          threshold: CONFIDENCE_THRESHOLD,
          results: null
        };
      }

      const plantInfo = mapping[classIdx];
      if (!plantInfo) {
        console.warn(`Class ${classIdx} not found in mapping`);
        continue;
      }

      results.push({
        _id: `student-${i}`,
        plant: {
          _id: plantInfo.id,
          latinName: plantInfo.latinName,
          commonName: plantInfo.commonName,
        },
        confidence: confidence,
        source: 'student-model',
        rank: i + 1
      });
    }

    return {
      success: true,
      results: results,
      model_version: CURRENT_MODEL_VERSION,
      confidence_threshold: CONFIDENCE_THRESHOLD
    };

  } catch (error) {
    console.error('Student recognition error:', error);
    throw error;
  }
}
```

#### 2.5 - Actualizar handleRecognize

```javascript
const handleRecognize = async (imageFile, captureMethod) => {
  try {
    setLoading(true);
    setError(null);

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = URL.createObjectURL(imageFile);

    img.onload = async () => {
      try {
        // Primero intentar con Student model (offline-first)
        let recognitionResults = await recognizeWithStudent(img);

        if (!recognitionResults.success) {
          // Si no es concluyente, usar PlantNet API como fallback
          console.log('Student model: not conclusive, falling back to PlantNet API');

          const apiResult = await axiosRequestFunctions.recognizePlant(
            imageFile,
            captureMethod
          );

          recognitionResults = {
            success: true,
            results: apiResult.recognition.results || [],
            source: 'plantnet-api',
            fallback_reason: 'low_confidence'
          };
        }

        setResults(recognitionResults);
        setCurrentImage(img.src);
        setLoading(false);
      } catch (err) {
        console.error('Recognition error:', err);
        setError(err.message);
        setLoading(false);
      }
    };

    img.onerror = () => {
      setError('Failed to load image');
      setLoading(false);
    };
  } catch (err) {
    console.error('Error in handleRecognize:', err);
    setError(err.message);
    setLoading(false);
  }
};
```

---

## Paso 3: Crear Endpoint Backend para Class Mapping

**Archivo:** `/home/skanndar/SynologyDrive/local/aplantida/aplantidaBack/routes/plants-router.js`

```javascript
router.get('/class-mapping', async (req, res) => {
  try {
    // Get all plants sorted by _id (must match training order)
    const plants = await Plant.find({})
      .select('_id latinName commonName')
      .sort('_id');  // CRITICAL: same order as during training

    // Create mapping: class_index → plant_info
    const mapping = {};
    plants.forEach((plant, index) => {
      mapping[index] = {
        id: plant._id.toString(),
        latinName: plant.latinName,
        commonName: plant.commonName
      };
    });

    // Cache for 30 days (class mapping doesn't change often)
    res.set('Cache-Control', 'public, max-age=2592000');
    res.json(mapping);
  } catch (err) {
    console.error('Error generating class mapping:', err);
    res.status(500).json({ error: err.message });
  }
});
```

**IMPORTANTE:** El orden de los plants debe ser **exactamente el mismo** que se usó durante el training. Si usaste un `class_mapping.json` durante el training, úsalo para validar.

---

## Paso 4: Actualizar Service Worker

**Archivo:** `/home/skanndar/SynologyDrive/local/aplantida/aplantidaFront/public/service-worker.js`

```javascript
// Import Workbox
importScripts('https://storage.googleapis.com/workbox-cdn/releases/6.5.4/workbox-sw.js');

// Precache del modelo student
workbox.precaching.precacheAndRoute([
  // Model files (actualiza según los shards reales generados)
  {url: '/models/student_v1.0/model.json', revision: 'v1.0'},
  {url: '/models/student_v1.0/group1-shard1of3.bin', revision: 'v1.0'},
  {url: '/models/student_v1.0/group1-shard2of3.bin', revision: 'v1.0'},
  {url: '/models/student_v1.0/group1-shard3of3.bin', revision: 'v1.0'},
  {url: '/models/student_v1.0/export_metadata.json', revision: 'v1.0'},

  // Add other app assets here
]);

// Cache strategy for model files
workbox.routing.registerRoute(
  ({request}) => request.url.includes('/models/'),
  new workbox.strategies.CacheFirst({
    cacheName: 'tf-models',
    plugins: [
      new workbox.expiration.ExpirationPlugin({
        maxAgeSeconds: 30 * 24 * 60 * 60,  // 30 days
        maxEntries: 10
      })
    ]
  })
);

// Cache strategy for class mapping
workbox.routing.registerRoute(
  ({url}) => url.pathname === '/api/plants/class-mapping',
  new workbox.strategies.CacheFirst({
    cacheName: 'plant-data',
    plugins: [
      new workbox.expiration.ExpirationPlugin({
        maxAgeSeconds: 30 * 24 * 60 * 60  // 30 days
      })
    ]
  })
);

// Handle model updates
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }

  if (event.data && event.data.type === 'CHECK_MODEL_UPDATE') {
    // Logic to check for new model versions
    fetch('/api/models/latest-version')
      .then(res => res.json())
      .then(data => {
        event.ports[0].postMessage({
          hasUpdate: data.version !== 'v1.0',
          newVersion: data.version
        });
      });
  }
});
```

---

## Paso 5: Script de Test en Navegador

Crea un archivo de test para verificar que el modelo carga correctamente:

**Archivo:** `/home/skanndar/SynologyDrive/local/aplantida/aplantidaFront/public/test-student-model.html`

```html
<!DOCTYPE html>
<html>
<head>
  <title>Test Student Model</title>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.4.0"></script>
</head>
<body>
  <h1>Student Model Test</h1>
  <div id="status">Loading...</div>
  <canvas id="canvas" width="224" height="224" style="display:none;"></canvas>

  <script>
    async function testModel() {
      const statusDiv = document.getElementById('status');

      try {
        // 1. Set backend
        await tf.setBackend('webgl');
        statusDiv.innerHTML += '<br>✓ Backend: webgl';

        // 2. Load model
        statusDiv.innerHTML += '<br>Loading model...';
        const model = await tf.loadGraphModel('/models/student_v1.0/model.json');
        statusDiv.innerHTML += '<br>✓ Model loaded';

        // 3. Get model info
        statusDiv.innerHTML += `<br>Model inputs: ${model.inputs.length}`;
        statusDiv.innerHTML += `<br>Model outputs: ${model.outputs.length}`;
        statusDiv.innerHTML += `<br>Input shape: ${model.inputs[0].shape}`;
        statusDiv.innerHTML += `<br>Output shape: ${model.outputs[0].shape}`;

        // 4. Test inference with random data
        const dummyInput = tf.randomNormal([1, 224, 224, 3]);
        statusDiv.innerHTML += '<br>Running test inference...';

        const output = model.predict(dummyInput);
        const probs = tf.softmax(output);
        const topK = tf.topk(probs, 5);

        const values = await topK.values.data();
        const indices = await topK.indices.data();

        statusDiv.innerHTML += '<br><br><b>Test Prediction (random input):</b>';
        for (let i = 0; i < 5; i++) {
          statusDiv.innerHTML += `<br>${i+1}. Class ${indices[i]}: ${(values[i] * 100).toFixed(2)}%`;
        }

        // Clean up
        dummyInput.dispose();
        output.dispose();
        probs.dispose();
        topK.values.dispose();
        topK.indices.dispose();

        statusDiv.innerHTML += '<br><br>✓ All tests passed!';

      } catch (error) {
        statusDiv.innerHTML += `<br><br>❌ Error: ${error.message}`;
        console.error(error);
      }
    }

    testModel();
  </script>
</body>
</html>
```

Accede a `http://localhost:3000/test-student-model.html` (ajusta el puerto) para verificar que el modelo se carga correctamente.

---

## Checklist de Deploy

- [ ] Model convertido con `tensorflowjs_converter` (ver CONVERSION_STATUS.md)
- [ ] Archivos copiados a `aplantidaFront/public/models/student_v1.0/`
- [ ] `PlantRecognition/index.js` actualizado con código student
- [ ] Backend endpoint `/api/plants/class-mapping` implementado
- [ ] Service Worker actualizado con precaché de model files
- [ ] Test HTML ejecutado y pasado
- [ ] Class mapping verificado (orden correcto)
- [ ] Threshold 0.62 confirmado en código
- [ ] Fallback a PlantNet API funcional cuando confidence < 0.62

---

## Notas Importantes

1. **Orden de clases:** El orden de plants en el class mapping DEBE coincidir con el orden usado durante training. Si no coincide, las predicciones serán incorrectas.

2. **Threshold calibrado:** No cambies el threshold de 0.62 sin re-calibrar. Este valor está optimizado para 94.5% accuracy y 80.1% coverage.

3. **Fallback a PlantNet:** El código está diseñado para usar PlantNet API cuando el student model tiene confianza < 0.62. Esto asegura que siempre haya una respuesta.

4. **Cache de modelo:** El modelo se cachea en IndexedDB para acceso offline. La primera carga descarga ~50-70MB, las siguientes son instantáneas.

5. **Actualización de modelo:** Para deployar una nueva versión (v1.1, etc.), actualiza `CURRENT_MODEL_VERSION` y agrega las nuevas URLs en `MODEL_URLS`. El Service Worker debe actualizarse también.

---

## Troubleshooting

**Modelo no carga:**
- Verifica que `model.json` existe en `/public/models/student_v1.0/`
- Verifica CORS si estás usando CDN
- Revisa Network tab en DevTools (F12)

**Predicciones incorrectas:**
- Verifica el orden del class mapping
- Verifica que la normalización es [0,1] (divide por 255)
- Verifica el tamaño de entrada (224x224)

**"No conclusive" en todas las predicciones:**
- Verifica que temperature scaling está aplicado (T=2.0)
- Verifica que el threshold es 0.62 (no 0.75 u otro valor)
- Verifica que el modelo es el calibrado (`best_model_temp.pt`)

**Cache no funciona:**
- Verifica quota de IndexedDB (Chrome: 50MB, Firefox: 1GB)
- Verifica Service Worker está registrado
- Limpia cache y vuelve a intentar (Settings → Clear browsing data)
