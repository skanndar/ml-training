#!/bin/bash
# Script para convertir SavedModel a TF.js usando @tensorflow/tfjs-converter (Node.js)

set -e

PROJECT_ROOT="/home/skanndar/SynologyDrive/local/aplantida/ml-training"
SAVED_MODEL="$PROJECT_ROOT/dist/models/student_v1_fp16_manual/saved_model"
OUTPUT_DIR="$PROJECT_ROOT/dist/models/student_v1_fp16"

echo "========================================="
echo "Conversión SavedModel → TF.js (Node.js)"
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

# Verificar Node.js
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js no encontrado. Instala Node.js >= 18"
    exit 1
fi

NODE_VERSION=$(node --version)
echo "✓ Node.js: $NODE_VERSION"
echo ""

# Instalar @tensorflow/tfjs-converter globalmente (si no existe)
echo "Verificando @tensorflow/tfjs-converter..."
if ! command -v tensorflowjs_converter &> /dev/null; then
    echo "Instalando @tensorflow/tfjs-converter globalmente..."
    npm install -g @tensorflow/tfjs-converter
    echo "✓ Convertidor instalado"
else
    echo "✓ Convertidor ya instalado"
fi
echo ""

# Limpiar output dir si existe
if [ -d "$OUTPUT_DIR" ]; then
    echo "Limpiando directorio de salida existente..."
    rm -rf "$OUTPUT_DIR"
fi

mkdir -p "$OUTPUT_DIR"

# Convertir a TF.js con cuantización FP16
echo "Ejecutando tensorflowjs_converter..."
echo "  Input:  $SAVED_MODEL"
echo "  Output: $OUTPUT_DIR"
echo "  Quantization: float16"
echo ""

tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    --quantize_float16='*' \
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

# Actualizar metadata con rutas TF.js
if [ -f "$OUTPUT_DIR/export_metadata.json" ]; then
    echo "✓ Actualizando metadata con rutas TF.js"

    # Actualizar campo tfjs_model en metadata usando Python
    python3 << 'PYTHON_SCRIPT'
import json
from pathlib import Path

metadata_path = Path("dist/models/student_v1_fp16/export_metadata.json")
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# Actualizar campo de export pipeline
metadata["export_pipeline"]["tfjs_model"] = "dist/models/student_v1_fp16/model.json"
metadata["export_pipeline"]["tfjs_conversion_method"] = "Node.js @tensorflow/tfjs-converter"

with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print("Metadata actualizado")
PYTHON_SCRIPT
fi

echo ""
echo "========================================="
echo "Verificación del modelo"
echo "========================================="
echo ""

# Verificar que model.json existe y es válido
if [ -f "$OUTPUT_DIR/model.json" ]; then
    echo "✓ model.json generado correctamente"

    # Mostrar información del modelo
    echo ""
    echo "Contenido de model.json (primeras 20 líneas):"
    head -20 "$OUTPUT_DIR/model.json"
    echo ""

    # Contar shards
    SHARD_COUNT=$(ls -1 "$OUTPUT_DIR"/*.bin 2>/dev/null | wc -l)
    echo "Shards generados: $SHARD_COUNT"

    if [ $SHARD_COUNT -gt 0 ]; then
        echo ""
        echo "Detalles de shards:"
        ls -lh "$OUTPUT_DIR"/*.bin
    fi
else
    echo "ERROR: model.json no fue generado"
    exit 1
fi

echo ""
echo "========================================="
echo "Siguiente paso:"
echo "========================================="
echo ""
echo "1. Verifica model.json en: $OUTPUT_DIR/model.json"
echo "2. Copia el directorio completo a aplantidaFront/public/models/"
echo "   cp -r $OUTPUT_DIR /ruta/a/aplantidaFront/public/models/student_v1.0"
echo ""
echo "3. Actualiza PlantRecognition/index.js:"
echo "   const CONFIDENCE_THRESHOLD = 0.62;"
echo "   const MODEL_URLS = { 'v1.0': '/models/student_v1.0/model.json' };"
echo ""
echo "4. Actualiza Service Worker para precachear:"
echo "   - /models/student_v1.0/model.json"
echo "   - /models/student_v1.0/group1-shard*.bin"
echo ""
echo "5. Test en navegador:"
echo "   const model = await tf.loadGraphModel('/models/student_v1.0/model.json');"
echo ""
