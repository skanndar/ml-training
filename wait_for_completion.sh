#!/bin/bash
# Wait for training to complete and show status

HISTORY_FILE="/media/skanndar/2TB1/aplantida-ml/results/teacher_global_v1/training_history.json"
CHECKPOINT_DIR="/media/skanndar/2TB1/aplantida-ml/checkpoints/teacher_global"

echo "=========================================="
echo "  WAITING FOR TRAINING COMPLETION"
echo "=========================================="
echo ""
echo "This will check for:"
echo "1. When training_history.json is created"
echo "2. When all 15 epochs are saved"
echo ""

while true; do
    EPOCH_COUNT=$(ls -1 $CHECKPOINT_DIR/checkpoint_epoch_*.pt 2>/dev/null | wc -l)

    if [ -f "$HISTORY_FILE" ] && [ -s "$HISTORY_FILE" ]; then
        echo ""
        echo "✅ TRAINING COMPLETE!"
        echo "---"
        echo "File: $HISTORY_FILE"
        echo "Size: $(du -h $HISTORY_FILE | cut -f1)"
        echo ""
        echo "Content preview:"
        cat "$HISTORY_FILE" | python3 -m json.tool | head -30
        echo ""
        echo "First epoch results:"
        echo "  Train Loss: $(cat $HISTORY_FILE | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d[\"train_loss\"][0])')"
        echo "  Train Top-1: $(cat $HISTORY_FILE | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d[\"train_top1_acc\"][0])' 2>/dev/null || echo 'N/A')%"
        echo ""
        echo "Completed epochs: $EPOCH_COUNT/15"
        exit 0
    else
        echo -ne "\r⏳ Waiting... Completed epochs: $EPOCH_COUNT/15"
        sleep 30
    fi
done
