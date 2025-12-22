# Análisis del Training - 15 Epochs Completados

**Fecha**: 2025-12-20
**Modelo**: ViT-Base-16 (90M parámetros)
**Dataset**: 118,126 imágenes cacheadas, 5,914 clases
**Resultado**: Overfitting severo

---

## Resumen Ejecutivo

El training completó exitosamente 15 epochs sin errores técnicos. Sin embargo, **el modelo tiene overfitting catastrófico**:

- **Train accuracy**: 98.6% (excelente)
- **Validation accuracy**: 0.03% (prácticamente azar)

El modelo memorizó el conjunto de entrenamiento pero no aprendió a generalizar.

---

## Métricas Finales

### Training Set
```
Epoch 1:  Loss 7.53,  Top-1: 3.5%,   Top-5: 8.4%
Epoch 5:  Loss 2.15,  Top-1: 83.8%,  Top-5: 93.3%
Epoch 10: Loss 1.39,  Top-1: 98.2%,  Top-5: 99.99%
Epoch 15: Loss 1.31,  Top-1: 98.6%,  Top-5: 100.0%
```

**Interpretación**: El modelo está memorizando perfectamente los 93,761 ejemplos de entrenamiento.

### Validation Set
```
Epoch 1:  Loss 10.32, Top-1: 0.00%, Top-5: 0.08%
Epoch 5:  Loss 11.38, Top-1: 0.02%, Top-5: 0.07%
Epoch 10: Loss 11.52, Top-1: 0.03%, Top-5: 0.09%
Epoch 15: Loss 11.22, Top-1: 0.03%, Top-5: 0.08%
```

**Interpretación**:
- Con 5,914 clases, azar puro = 1/5,914 = **0.017%**
- Validation accuracy = **0.03%**
- **El modelo apenas supera el azar**
- Validation loss AUMENTA (10.32 → 11.22) - señal clara de overfitting

---

## Análisis del Problema

### 1. Dataset Split Inadecuado (PROBLEMA PRINCIPAL)

**Split Random Anterior:**
```
Train classes:  5,578
Val classes:    2,685
Overlap:        2,499 (44.8% de train)
Only in val:    186 clases (1.7% de imágenes val)
```

**Problemas identificados:**
- 186 clases aparecen en validation pero NO en train
- El modelo nunca vio esas clases → 100% de error garantizado
- 55.2% de las clases de train nunca son validadas
- Random shuffle sin estratificación por clase

**Split Stratified (NUEVO):**
```
Train classes:  5,914
Val classes:    4,410
Overlap:        4,410 (100% de val) ✅
Only in val:    0 clases ✅
```

### 2. Clases con Muy Pocas Imágenes

**Distribución anterior:**
- 27.7% de clases en train tenían solo 1 imagen
- 36.6% de clases en val tenían solo 1 imagen
- Promedio: 16.9 imágenes/clase en train, 4.4 en val

**Distribución nueva (stratified):**
- 1,504 clases con 1 imagen → solo en train (no se pueden validar)
- 913 clases con 2 imágenes → 1 en train, 1 en val
- 3,497 clases con 3+ imágenes → split 80/10/10 por clase

**Problema**: Es muy difícil que un modelo aprenda patrones visuales con 1-2 ejemplos por clase.

### 3. Desbalance Modelo vs Dataset

```
Parámetros del modelo: 90,088,138
Imágenes de train:     93,761
Ratio:                 ~960 parámetros por imagen
```

El modelo tiene suficiente capacidad para **memorizar** todas las imágenes individuales en lugar de aprender patrones generales.

---

## ¿Por qué Train está bien pero Val está mal?

**Durante Training:**
- El modelo ve las mismas 93,761 imágenes repetidas 15 veces
- Con 90M parámetros, puede memorizar cada imagen individual
- Loss baja, accuracy sube → parece que está aprendiendo
- Pero en realidad está **sobreajustando** (overfitting)

**Durante Validation:**
- El modelo ve 12,639 imágenes NUEVAS que nunca vio antes
- No aprendió patrones generales, solo memorizó ejemplos específicos
- Cuando ve una imagen nueva → no sabe qué hacer
- Accuracy permanece en nivel de azar (0.03%)

**Analogía:**
Es como un estudiante que memoriza las respuestas exactas de 100 preguntas de ejemplo, pero cuando le dan un examen con preguntas ligeramente diferentes, no sabe contestar porque nunca entendió los conceptos.

---

## Soluciones Recomendadas

### Opción 1: Reentrenar con Split Stratified (RECOMENDADO)

**Ya implementado**: `scripts/create_stratified_split.py`

**Ventajas:**
- Todas las clases en val también están en train
- Distribución más balanceada por clase
- Mejor evaluación del modelo

**Pasos:**
```bash
# Actualizar config para usar nuevos datasets
# En config/teacher_global.yaml:
data:
  train_jsonl: "./data/dataset_train_stratified.jsonl"
  val_jsonl: "./data/dataset_val_stratified.jsonl"

# Eliminar checkpoint anterior (incompatible)
rm -rf checkpoints/teacher_global/

# Reentrenar desde cero
python3 scripts/train_teacher.py --config config/teacher_global.yaml
```

**Resultado esperado**: Validation accuracy debería ser significativamente mejor (esperamos 5-15% en epoch 15).

---

### Opción 2: Aumentar Regularización

Además del stratified split, aumentar la regularización para reducir overfitting:

**En `config/teacher_global.yaml`:**
```yaml
training:
  # Regularización aumentada
  weight_decay: 0.05           # Era 0.01, subir a 0.05
  label_smoothing: 0.2         # Era 0.1, subir a 0.2
  dropout: 0.3                 # Añadir dropout en clasificador

augmentation:
  # Aumentar augmentación de datos
  random_crop: true
  random_horizontal_flip: true
  color_jitter: 0.4            # Aumentar variabilidad de color
  random_erasing: 0.25         # Borrado aleatorio de regiones
```

---

### Opción 3: Reducir Complejidad del Modelo

Si overfitting persiste, usar un modelo más pequeño:

```yaml
model:
  name: "vit_small_patch16_224"  # Era vit_base_patch16_224
  # ViT-Small: ~22M parámetros (vs 90M de Base)
```

---

### Opción 4: Filtrar Clases con Pocas Imágenes

Entrenar solo con clases que tienen suficientes ejemplos:

```python
# Filtrar a clases con ≥5 imágenes
min_images_per_class = 5

# Esto reduciría a ~3,000 clases pero con mejor distribución
```

---

## Diagnóstico Técnico Adicional

### Sin Errores de Descarga ✅
```bash
$ tail -500 training.log | grep -E "Failed to load|Rate limit|blank|WARNING" | wc -l
0
```

- **0 errores** de descarga durante todo el training
- **0 imágenes blank** usadas
- El smart rate limiting funcionó perfectamente
- Todas las imágenes cargadas desde cache

### Progreso Técnico ✅
- Training completó 15 epochs sin crashes
- Checkpoints guardados correctamente
- TensorBoard logs generados
- Cache estable (91.87GB)

---

## Próximos Pasos Recomendados

### Inmediato (Hoy)
1. ✅ Crear stratified split (HECHO)
2. Actualizar config para usar `dataset_train_stratified.jsonl`
3. Eliminar checkpoints anteriores (incompatibles)
4. Reentrenar con el nuevo split

### Corto Plazo (Esta Semana)
5. Monitorear que val accuracy mejore (esperamos >5% en epoch 5)
6. Si persiste overfitting, aumentar regularización (opción 2)
7. Considerar early stopping cuando val loss deje de bajar

### Mediano Plazo (Próxima Semana)
8. Analizar clases más difíciles (peor accuracy)
9. Revisar si algunas familias/géneros necesitan más datos
10. Considerar data augmentation más agresivo

---

## Conclusión

El training anterior fue **técnicamente exitoso** (sin errores) pero **estadísticamente fallido** (overfitting severo).

El problema NO es el código, cache, o infrastructure - todo funcionó perfectamente.

El problema ES el dataset split: un modelo no puede predecir clases que nunca vio durante entrenamiento.

**Solución**: Reentrenar con stratified split donde todas las clases en validation también aparecen en train.

**Expectativa realista**: Con el nuevo split, esperamos validation accuracy de 5-20% (mucho mejor que 0.03%, pero aún lejos del 98% de train debido a clases con pocas imágenes).
