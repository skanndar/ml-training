#!/bin/bash
# Real-time training monitor

echo "================================"
echo "  APLANTIDA TRAINING MONITOR"
echo "================================"
echo ""

# Check if training is running
if ! pgrep -f "python3 scripts/train_teacher.py" > /dev/null; then
    echo "❌ No training process running"
    exit 1
fi

echo "✅ Training is RUNNING"
echo ""

# Show process info
MAIN_PID=$(ps aux | grep "python3 scripts/train_teacher.py" | grep -v grep | head -1 | awk '{print $2}')
echo "Main Process PID: $MAIN_PID"
echo ""

# GPU Status
echo "📊 GPU Status:"
echo "---"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader | head -1
echo ""

# Cache Status
echo "💾 Cache Status:"
echo "---"
CACHE_SIZE=$(du -sh /media/skanndar/2TB1/aplantida-ml/image_cache 2>/dev/null | cut -f1)
echo "Cache size: $CACHE_SIZE / 100GB"
echo ""

# Checkpoint Status
echo "✓ Completed Checkpoints:"
echo "---"
EPOCH_COUNT=$(ls -1 /media/skanndar/2TB1/aplantida-ml/checkpoints/teacher_global/checkpoint_epoch_*.pt 2>/dev/null | wc -l)
echo "Epochs saved: $EPOCH_COUNT/15"
if [ $EPOCH_COUNT -gt 0 ]; then
    LATEST_EPOCH=$(ls -1 /media/skanndar/2TB1/aplantida-ml/checkpoints/teacher_global/checkpoint_epoch_*.pt 2>/dev/null | tail -1 | grep -o "epoch_[0-9]*" | cut -d_ -f2)
    echo "Latest epoch: $LATEST_EPOCH"
fi
echo ""

# Results Status
echo "📈 Results Files:"
echo "---"
if [ -f "/media/skanndar/2TB1/aplantida-ml/results/teacher_global_v1/training_history.json" ]; then
    echo "✅ training_history.json: Created"
else
    echo "⏳ training_history.json: Not yet (still training)"
fi

# PNG files
PNG_COUNT=$(ls -1 /media/skanndar/2TB1/aplantida-ml/results/teacher_global_v1/*.png 2>/dev/null | wc -l)
echo "📊 PNG plots: $PNG_COUNT/4"

echo ""
echo "================================"
echo "Last updated: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================"
