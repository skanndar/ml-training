#!/bin/bash
# Script para convertir SavedModel a TF.js usando un entorno limpio con TF 2.13.x

set -e

PROJECT_ROOT="/home/skanndar/SynologyDrive/local/aplantida/ml-training"
TFJS_ENV="$PROJECT_ROOT/tfjs-env"
SAVED_MODEL="$PROJECT_ROOT/dist/models/student_v1_fp16_manual/saved_model"
OUTPUT_DIR="$PROJECT_ROOT/dist/models/student_v1_fp16"

echo "========================================="
echo "Conversión SavedModel → TF.js"
echo "========================================="
echo ""

# Verificar que existe el SavedModel
if [ ! -d "$SAVED_MODEL" ]; then
    echo "ERROR: SavedModel no encontrado en $SAVED_MODEL"
    echo "Ejecuta primero scripts/export_to_tfjs.py"
    exit 1
fi

echo "✓ SavedModel encontrado: $SAVED_MODEL"
echo ""

# Crear entorno virtual si no existe
if [ ! -d "$TFJS_ENV" ]; then
    echo "Creando entorno virtual para TF.js converter..."
    python3 -m venv "$TFJS_ENV"
    echo "✓ Entorno virtual creado"
else
    echo "✓ Entorno virtual ya existe"
fi

# Activar entorno
source "$TFJS_ENV/bin/activate"

# Instalar dependencias compatibles (TF 2.13.x)
echo ""
echo "Instalando dependencias compatibles..."
echo "  - TensorFlow 2.13.1"
echo "  - TensorFlowJS 4.22.0"
echo "  - TensorFlow Decision Forests 1.5.0"
echo "  - YDF 0.13.0"
echo ""

pip install --quiet --upgrade pip
pip install --quiet \
    "tensorflow==2.13.1" \
    "tensorflowjs==4.22.0" \
    "tensorflow-decision-forests==1.5.0" \
    "ydf==0.13.0" \
    "tensorflow-hub==0.16.1" \
    "tensorflow-addons==0.21.0"

echo "✓ Dependencias instaladas"
echo ""

# Convertir a TF.js
echo "Ejecutando tensorflowjs_converter..."
echo "  Input:  $SAVED_MODEL"
echo "  Output: $OUTPUT_DIR"
echo ""

tensorflowjs_converter \
    --input_format tf_saved_model \
    --output_format tfjs_graph_model \
    --quantize_float16 '*' \
    "$SAVED_MODEL" \
    "$OUTPUT_DIR"

echo ""
echo "========================================="
echo "✓ Conversión completada exitosamente"
echo "========================================="
echo ""

# Mostrar archivos generados
echo "Archivos generados:"
ls -lh "$OUTPUT_DIR"
echo ""

# Calcular tamaño total
TOTAL_SIZE=$(du -sh "$OUTPUT_DIR" | cut -f1)
echo "Tamaño total del modelo: $TOTAL_SIZE"
echo ""

# Copiar metadata
if [ -f "$PROJECT_ROOT/dist/models/student_v1_fp16_manual/export_metadata.json" ]; then
    cp "$PROJECT_ROOT/dist/models/student_v1_fp16_manual/export_metadata.json" "$OUTPUT_DIR/"
    echo "✓ export_metadata.json copiado"
fi

echo ""
echo "========================================="
echo "Siguiente paso:"
echo "========================================="
echo ""
echo "1. Verifica model.json en: $OUTPUT_DIR/model.json"
echo "2. Copia el directorio completo a aplantidaFront/public/models/"
echo "3. Actualiza PlantRecognition/index.js con CONFIDENCE_THRESHOLD=0.62"
echo "4. Actualiza Service Worker para precachear los nuevos shards"
echo ""

deactivate
