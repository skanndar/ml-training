#!/bin/bash

# Real-time training monitoring dashboard
# Shows progress, GPU stats, and training metrics

CHECKPOINT_DIR="/media/skanndar/2TB1/aplantida-ml/checkpoints/teacher_global"
HISTORY_FILE="/media/skanndar/2TB1/aplantida-ml/results/teacher_global_v1/training_history.json"
LOG_FILE="/home/skanndar/SynologyDrive/local/aplantida/ml-training/training.log"

clear

echo "============================================================================="
echo "  APLANTIDA ViT-Base Teacher Training - Real-Time Monitor"
echo "============================================================================="
echo ""

# Get PID of main training process
MAIN_PID=$(ps aux | grep "python3 scripts/train_teacher.py" | grep -v grep | head -1 | awk '{print $2}')

if [ -z "$MAIN_PID" ]; then
    echo "❌ No training process running"
    exit 1
fi

echo "✅ Training ACTIVE (PID: $MAIN_PID)"
START_TIME=$(ps -p $MAIN_PID -o lstart=)
echo "   Started: $START_TIME"
echo ""

# GPU Status
echo "📊 GPU Status:"
echo "─────────────────────────────────────────────────────────────────────────"
nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader | while IFS=',' read -r GPU UTIL_GPU UTIL_MEM MEM_USED MEM_TOTAL TEMP; do
    echo "  GPU: $GPU"
    echo "    Utilization: $UTIL_GPU GPU, $UTIL_MEM Memory"
    echo "    Memory: ${MEM_USED} / ${MEM_TOTAL}"
    echo "    Temperature: $TEMP°C"
done
echo ""

# Checkpoint Progress
echo "✓ Training Progress:"
echo "─────────────────────────────────────────────────────────────────────────"
EPOCH_COUNT=$(ls -1 "$CHECKPOINT_DIR/checkpoint_epoch_*.pt" 2>/dev/null | wc -l)
echo "  Epochs completed: $EPOCH_COUNT/15"

if [ $EPOCH_COUNT -gt 0 ]; then
    LATEST_EPOCH=$(ls -1 "$CHECKPOINT_DIR/checkpoint_epoch_*.pt" 2>/dev/null | tail -1 | grep -o "epoch_[0-9]*" | cut -d_ -f2)
    LATEST_SIZE=$(ls -lh "$CHECKPOINT_DIR/checkpoint_epoch_${LATEST_EPOCH}.pt" 2>/dev/null | awk '{print $5}')
    LATEST_TIME=$(ls -lh "$CHECKPOINT_DIR/checkpoint_epoch_${LATEST_EPOCH}.pt" 2>/dev/null | awk '{print $6, $7, $8}')
    echo "  Latest: Epoch $LATEST_EPOCH ($LATEST_SIZE) at $LATEST_TIME"
fi
echo ""

# Latest Training Metrics
echo "📈 Current Metrics (Last batch):"
echo "─────────────────────────────────────────────────────────────────────────"
LATEST_LOG=$(tail -5 "$LOG_FILE" | grep -E "\[.*it/s.*loss" | tail -1)
if [ ! -z "$LATEST_LOG" ]; then
    echo "  $LATEST_LOG"
else
    echo "  Loading training data..."
fi
echo ""

# Cache Status
echo "💾 Cache Status:"
echo "─────────────────────────────────────────────────────────────────────────"
CACHE_SIZE=$(du -sh /media/skanndar/2TB1/aplantida-ml/image_cache 2>/dev/null | awk '{print $1}')
echo "  Cache size: $CACHE_SIZE / 100GB"
echo ""

# Storage Status
echo "💿 Storage:"
echo "─────────────────────────────────────────────────────────────────────────"
df -h /media/skanndar/2TB1 | tail -1 | awk '{printf "  2TB Drive: %s used / %s total (%s full)\n", $3, $2, $5}'
df -h /home/skanndar/SynologyDrive/local | tail -1 | awk '{printf "  Primary:   %s used / %s total (%s full)\n", $3, $2, $5}'
echo ""

# Results Status
echo "📋 Results:"
echo "─────────────────────────────────────────────────────────────────────────"
if [ -f "$HISTORY_FILE" ] && [ -s "$HISTORY_FILE" ]; then
    echo "  ✅ training_history.json: CREATED (Training completed)"
else
    echo "  ⏳ training_history.json: Not yet (Still training)"
fi
echo ""

# Timing
echo "⏱️  Last update: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================================="
echo ""
echo "Watch command to auto-refresh every 30 seconds:"
echo "  watch -n 30 'bash /home/skanndar/SynologyDrive/local/aplantida/ml-training/monitor_training.sh'"
echo ""
