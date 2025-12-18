#!/bin/bash
# Real-time training monitor

echo "🔄 Monitoring training process..."
echo ""

# Main monitoring loop
while true; do
    clear

    echo "════════════════════════════════════════════════════════════"
    echo "  APLANTIDA TRAINING MONITOR"
    echo "════════════════════════════════════════════════════════════"
    echo ""

    # 1. Check if process is running
    PID=$(ps aux | grep "train_teacher.py" | grep -v grep | awk '{print $2}' | head -1)

    if [ -z "$PID" ]; then
        echo "❌ Training NOT running"
        echo ""
        echo "Check log file: training.log"
        tail -20 training.log
        break
    else
        echo "✅ Training RUNNING (PID: $PID)"
        echo ""
    fi

    # 2. Training log - show last lines
    echo "📊 Latest training output:"
    echo "─────────────────────────────────────────────────────────────"
    tail -5 training.log | grep -E "Epoch|Loss|Top|saved" || echo "Loading..."
    echo ""

    # 3. GPU Status
    echo "🖥️  GPU Status:"
    echo "─────────────────────────────────────────────────────────────"
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader
    echo ""

    # 4. Checkpoints
    echo "✓ Checkpoints saved:"
    echo "─────────────────────────────────────────────────────────────"
    CHECKPOINT_COUNT=$(ls -1 /media/skanndar/2TB1/aplantida-ml/checkpoints/teacher_global/checkpoint_epoch_*.pt 2>/dev/null | wc -l)
    echo "Completed epochs: $CHECKPOINT_COUNT/15"

    if [ $CHECKPOINT_COUNT -gt 0 ]; then
        LATEST=$(ls -1 /media/skanndar/2TB1/aplantida-ml/checkpoints/teacher_global/checkpoint_epoch_*.pt 2>/dev/null | tail -1 | xargs basename | sed 's/.*_//' | sed 's/.pt//')
        echo "Latest epoch: $LATEST"
        echo "Last checkpoint saved: $(ls -lh /media/skanndar/2TB1/aplantida-ml/checkpoints/teacher_global/checkpoint_epoch_${LATEST}.pt 2>/dev/null | awk '{print $6, $7, $8}')"
    fi
    echo ""

    # 5. Results
    echo "📈 Results:"
    echo "─────────────────────────────────────────────────────────────"
    if [ -f "/media/skanndar/2TB1/aplantida-ml/results/teacher_global_v1/training_history.json" ] && [ -s "/media/skanndar/2TB1/aplantida-ml/results/teacher_global_v1/training_history.json" ]; then
        echo "✅ training_history.json: CREATED"
        echo "   File size: $(du -h /media/skanndar/2TB1/aplantida-ml/results/teacher_global_v1/training_history.json | awk '{print $1}')"
        echo ""
        echo "🎉 TRAINING COMPLETED!"
        break
    else
        echo "⏳ training_history.json: Not yet (still training)"
    fi
    echo ""

    # 6. Cache status
    echo "💾 Cache Status:"
    echo "─────────────────────────────────────────────────────────────"
    CACHE_SIZE=$(du -sh /media/skanndar/2TB1/aplantida-ml/image_cache 2>/dev/null | awk '{print $1}')
    echo "Cache size: $CACHE_SIZE / 100GB"
    echo ""

    # 7. Storage
    echo "💿 Storage:"
    echo "─────────────────────────────────────────────────────────────"
    df -h /media/skanndar/2TB1 | tail -1 | awk '{printf "2TB Drive: %s used / %s total (%s full)\n", $3, $2, $5}'
    df -h /home/skanndar/SynologyDrive/local | tail -1 | awk '{printf "Primary:   %s used / %s total (%s full)\n", $3, $2, $5}'
    echo ""

    # 8. Timing
    echo "⏱️  Last update: $(date '+%H:%M:%S')"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    echo ""

    # Wait 30 seconds before refresh
    sleep 30
done
