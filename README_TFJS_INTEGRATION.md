# Guía Rápida: Integración TF.js

**Estado:** ✅ Modelo convertido y verificado
**Próximo paso:** Integración en PWA

---

## 🎯 Inicio Rápido

### 1. Verificar el Modelo

```bash
node scripts/verify_tfjs_model.js
```

**Output esperado:**
- ✅ Modelo cargado exitosamente
- ✅ Input: [batch, 3, 224, 224]
- ✅ Inferencia exitosa en ~95ms
- ✅ Tamaño: 50.54 MB

### 2. Copiar al Frontend

```bash
# Ajusta la ruta según tu proyecto
cp -r dist/models/student_v1_fp16 /ruta/a/aplantidaFront/public/models/student_v1.0
```

### 3. Código Básico de Integración

```javascript
// Cargar modelo
const model = await tf.loadGraphModel('/models/student_v1.0/model.json');

// Preprocessing
function preprocessImage(img) {
  return tf.tidy(() => {
    let tensor = tf.browser.fromPixels(img);
    tensor = tf.image.resizeBilinear(tensor, [224, 224]);
    tensor = tensor.div(255.0);
    const mean = tf.tensor1d([0.485, 0.456, 0.406]);
    const std = tf.tensor1d([0.229, 0.224, 0.225]);
    tensor = tensor.sub(mean).div(std);
    tensor = tensor.transpose([2, 0, 1]);  // HWC → CHW
    tensor = tensor.expandDims(0);         // Add batch
    return tensor;
  });
}

// Inferencia
function predict(model, imageTensor) {
  const logits = model.predict(imageTensor);
  const probs = tf.softmax(logits);
  const topK = tf.topk(probs, 5);
  return topK;
}

// Uso
const imageTensor = preprocessImage(imageElement);
const predictions = await predict(model, imageTensor);
const values = await predictions.values.data();

if (values[0] >= 0.62) {
  console.log('Predicción confiable:', values[0]);
} else {
  console.log('Usar fallback a PlantNet API');
}
```

---

## 📚 Documentación Completa

### Guías por Rol

| Si eres... | Lee esto primero |
|------------|------------------|
| **Frontend Developer** | [CONVERSION_COMPLETED.md](CONVERSION_COMPLETED.md) §2-4 |
| **DevOps** | [CONVERSION_COMPLETED.md](CONVERSION_COMPLETED.md) §3 (Service Worker) |
| **Product Manager** | [ESTADO_FINAL_24DIC2025.md](ESTADO_FINAL_24DIC2025.md) |
| **ML Engineer** | [EXPORT_TFJS_PWA.md](aplantida-ml-training/EXPORT_TFJS_PWA.md) |
| **Quiero entender threshold** | [EXPORT_TFJS_PWA.md §1.5](aplantida-ml-training/EXPORT_TFJS_PWA.md#15---decisión-del-threshold-de-confianza) |

### Archivos Clave

1. **[ESTADO_FINAL_24DIC2025.md](ESTADO_FINAL_24DIC2025.md)**
   - Resumen ejecutivo
   - Estado completo
   - Próximos pasos

2. **[CONVERSION_COMPLETED.md](CONVERSION_COMPLETED.md)**
   - Guía completa de integración
   - Código JavaScript completo
   - Service Worker config
   - HTML de test

3. **[FRONTEND_INTEGRATION.md](FRONTEND_INTEGRATION.md)**
   - Arquitectura frontend
   - Código listo para copiar/pegar
   - Manejo de errores
   - Cache strategies

4. **[GOOGLE_COLAB_CONVERSION.md](GOOGLE_COLAB_CONVERSION.md)**
   - Cómo se hizo la conversión
   - Resolución de problemas
   - Compatible con TF 2.17+

5. **[EXPORT_TFJS_PWA.md §1.5](aplantida-ml-training/EXPORT_TFJS_PWA.md#15---decisión-del-threshold-de-confianza)**
   - Justificación threshold 0.62
   - Análisis comparativo
   - Trade-offs explicados

---

## 🔧 Configuración

### Threshold de Confianza

**Valor:** 0.62

**Significado:**
- Predicciones con confidence >= 0.62 → Mostrar al usuario
- Predicciones con confidence < 0.62 → Fallback a PlantNet API

**Métricas:**
- Accuracy: 94.5% (en predicciones confiables)
- Coverage: 80.1% (80% de las veces no necesita internet)
- Fallback rate: 19.9% (solo 1 de cada 5 veces usa API)

**Por qué 0.62:**
- Optimizado para zonas rurales sin internet
- Mejor coverage que threshold 0.66 (+1.2%)
- Pérdida mínima de accuracy (-0.5%)
- Ver análisis completo en [EXPORT_TFJS_PWA.md §1.5](aplantida-ml-training/EXPORT_TFJS_PWA.md#15---decisión-del-threshold-de-confianza)

### Input Format

- **Tamaño:** 224x224 pixels
- **Formato:** NCHW (channels first)
- **Tipo:** float32
- **Rango:** Normalizado con ImageNet stats
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]

### Output Format

- **Shape:** [batch, 8587]
- **Tipo:** float32 (logits)
- **Procesamiento:** Aplicar `tf.softmax()` para probabilidades
- **Clases activas:** 7120 (filtrar del total de 8587)

---

## ⚠️ Notas Importantes

### 1. Número de Clases

El modelo tiene **8587 clases** (dataset completo de PlantNet).

**Acción requerida:**
- Filtrar solo las 7120 clases activas en el frontend
- Usar mapping de índices a especies
- Ver [CONVERSION_COMPLETED.md](CONVERSION_COMPLETED.md) para detalles

### 2. Formato NCHW

TensorFlow.js usa **NCHW** (channels first) por defecto para este modelo.

```javascript
// Correcto ✅
tensor.transpose([2, 0, 1])  // HWC → CHW

// Incorrecto ❌
// No hacer transpose (quedará en HWC)
```

### 3. Softmax Requerido

El modelo retorna **logits** (sin softmax).

```javascript
// Correcto ✅
const probs = tf.softmax(logits)

// Incorrecto ❌
// Usar logits directamente como probabilidades
```

---

## 🧪 Testing

### Test Básico en Node.js

```bash
node scripts/verify_tfjs_model.js
```

### Test en Navegador

1. Copiar [test.html](CONVERSION_COMPLETED.md#4-test-en-navegador) del apartado §4
2. Servir con: `python -m http.server 8000`
3. Abrir: http://localhost:8000/test.html
4. Upload imagen y verificar predicciones

### Test de Performance

```javascript
// Medir tiempo de inferencia
const start = performance.now();
const output = model.predict(tensor);
await output.data();  // Forzar ejecución
const time = performance.now() - start;
console.log(`Inferencia: ${time.toFixed(2)}ms`);
```

**Tiempos esperados:**
- **Desktop (CPU):** 50-150ms
- **Mobile (CPU):** 200-500ms
- **Mobile (WebGL):** 100-300ms

---

## 📊 Métricas de Monitoreo

### En Producción, Trackear:

1. **Accuracy efectivo**
   - % de predicciones correctas (con user feedback)
   - Comparar con 94.5% esperado

2. **Coverage real**
   - % de predicciones con confidence >= 0.62
   - Esperado: ~80%

3. **Fallback rate**
   - % de veces que se usa PlantNet API
   - Esperado: ~20%

4. **Latencia**
   - Tiempo de inferencia (p50, p95, p99)
   - Esperado: < 500ms en mobile

5. **Distribución de confianza**
   - Histograma de confidence scores
   - Identificar si threshold 0.62 sigue siendo óptimo

### Ajustar Threshold Dinámicamente

Si en producción observas:

- **Fallback rate > 30%** → Bajar threshold a 0.58-0.60
- **Accuracy < 92%** → Subir threshold a 0.64-0.66
- **User feedback negativo** → Analizar casos edge y ajustar

---

## 🚀 Deployment Checklist

- [ ] Modelo copiado a `/public/models/student_v1.0/`
- [ ] PlantRecognition.js actualizado
- [ ] Preprocessing correcto (NCHW + ImageNet norm)
- [ ] Softmax aplicado a logits
- [ ] Threshold 0.62 configurado
- [ ] Fallback a PlantNet implementado
- [ ] Service Worker configurado (precache 14 archivos)
- [ ] Test en navegador desktop
- [ ] Test en navegador mobile
- [ ] Mapping 8587→7120 clases verificado
- [ ] IndexedDB para cache de predicciones
- [ ] Métricas de monitoreo configuradas
- [ ] Logging de errores habilitado
- [ ] Deploy a staging
- [ ] Test end-to-end en staging
- [ ] Deploy a producción

---

## 🆘 Troubleshooting

### Error: "Model not found"

```javascript
// Verificar que el path es correcto
console.log('Loading from:', MODEL_URL);

// Verificar en Network tab (DevTools) que se carga model.json
```

### Error: "Shape mismatch"

```javascript
// Verificar que el tensor es NCHW
console.log('Input shape:', tensor.shape);  // Debe ser [1, 3, 224, 224]

// No [1, 224, 224, 3] (NHWC)
```

### Error: "Softmax required"

```javascript
// Logits pueden ser negativos
console.log('Logits:', await logits.data());  // Puede contener negativos

// Softmax convierte a probabilidades [0,1]
const probs = tf.softmax(logits);
console.log('Probs:', await probs.data());    // Todos entre 0 y 1, suman 1
```

### Performance lenta

```javascript
// Usar WebGL backend
await tf.setBackend('webgl');

// Verificar backend actual
console.log('Backend:', tf.getBackend());  // Debe ser 'webgl' en navegador

// CPU es más lento pero más compatible
```

---

## 📞 Contacto y Referencias

- **Documentación ML:** `aplantida-ml-training/`
- **Verificación modelo:** `node scripts/verify_tfjs_model.js`
- **Threshold analysis:** [EXPORT_TFJS_PWA.md §1.5](aplantida-ml-training/EXPORT_TFJS_PWA.md#15---decisión-del-threshold-de-confianza)

---

**Última actualización:** 24 de diciembre de 2025, 19:25
**Versión modelo:** v1.0
**Estado:** ✅ Listo para integración
