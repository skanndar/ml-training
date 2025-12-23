#!/bin/bash
# Launch Teacher B (EU_SW) training with 384px + smart crop + stratified splits

set -euo pipefail

REGION="EU_SW"
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

CONFIG_FILE="config/teacher_regional.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "❌ No se encuentra $CONFIG_FILE"
  exit 1
fi

if [ ! -d "venv" ]; then
  echo "❌ Virtualenv 'venv' no existe. Ejecuta 'python -m venv venv && source venv/bin/activate && pip install -r requirements.txt'"
  exit 1
fi

source venv/bin/activate

echo "✅ Virtualenv activado"

# Detect cache dir desde config/paths.yaml o fallback
CACHE_DIR=$(python3 - <<'PY'
from pathlib import Path
cache = Path('./data/image_cache').resolve()
try:
    import yaml
    with open('config/paths.yaml', 'r', encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh) or {}
        data = cfg.get('data', {}) or {}
        candidate = data.get('cache_dir')
        if candidate:
            cache = Path(candidate).resolve()
except Exception:
    pass
print(cache)
PY
)

echo "🗃️  Cache dir: $CACHE_DIR"

DATA_PREFIX="./data/dataset_eu_sw"
TRAIN_FILE="${DATA_PREFIX}_train_stratified.jsonl"
VAL_FILE="${DATA_PREFIX}_val_stratified.jsonl"
TEST_FILE="${DATA_PREFIX}_test_stratified.jsonl"
STATS_FILE="${DATA_PREFIX}_stratified_stats.json"

if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$VAL_FILE" ]; then
  echo "⚠️  No se encuentran los splits estratificados EU_SW. Generándolos..."
  python3 scripts/create_stratified_split.py \
    --input ./data/dataset_raw.jsonl \
    --output-prefix "$DATA_PREFIX" \
    --region "$REGION" \
    --cache-dir "$CACHE_DIR"
  echo ""
fi

if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$VAL_FILE" ]; then
  echo "❌ No se generaron los splits regionales (revisa logs)."
  exit 1
fi

echo "✅ Splits regionales listos"

if [ -f "$STATS_FILE" ]; then
  echo "📊 Stats EU_SW:"
  python3 - <<PY
import json
with open('$STATS_FILE') as fh:
    stats = json.load(fh)
print(json.dumps(stats, indent=2))
PY
fi

# Backup checkpoints antiguos si existen
if [ -d "checkpoints/teacher_regional" ]; then
  BACKUP_DIR="checkpoints/teacher_regional_backup_$(date +%Y%m%d_%H%M%S)"
  echo "⚠️  Respaldo de checkpoints previos → $BACKUP_DIR"
  mv checkpoints/teacher_regional "$BACKUP_DIR"
fi

INIT_FROM=$(python3 - <<'PY'
import yaml, json
with open('config/teacher_regional.yaml', 'r') as fh:
    cfg = yaml.safe_load(fh)
print(cfg.get('model', {}).get('init_from', ''))
PY
)
if [ -n "$INIT_FROM" ] && [ ! -f "$INIT_FROM" ]; then
  echo "⚠️  Aviso: init_from apunta a '$INIT_FROM' pero no existe. Se usará el checkpoint de ImageNet."
fi

echo "==============================================================="
echo "  Iniciando entrenamiento Teacher Regional ($REGION)"
echo "==============================================================="
echo "Model: vit_base_patch16_384"
echo "Train file: $TRAIN_FILE"
echo "Val file:   $VAL_FILE"
echo "Cache dir:  $CACHE_DIR"
echo "Log file:   training_regional_384.log"
echo ""

nohup python3 scripts/train_teacher.py \
  --config "$CONFIG_FILE" \
  > training_regional_384.log 2>&1 &

TRAIN_PID=$!

echo "✅ Entrenamiento lanzado (PID $TRAIN_PID)"
echo "Monitoriza con: tail -f training_regional_384.log | grep -E 'Epoch|loss|top'"
echo "TensorBoard: tensorboard --logdir ./checkpoints/teacher_regional/logs"
echo ""
