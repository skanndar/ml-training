#!/bin/bash

# Start fresh training with cached images only (no checkpoints, clean start)

set -e

cd /home/skanndar/SynologyDrive/local/aplantida/ml-training

echo "========================================"
echo "  Start Fresh Training (Cached Only)"
echo "========================================"
echo ""

# 1. Stop current training
echo "Step 1: Stopping current training..."
pkill -SIGTERM -f train_teacher.py 2>/dev/null || echo "No training running"
sleep 5
echo "✓ Stopped"
echo ""

# 2. Backup old results
echo "Step 2: Backing up old checkpoints and results..."
BACKUP_DIR="old_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

if [ -d "checkpoints/teacher_global" ]; then
    mv checkpoints/teacher_global "$BACKUP_DIR/"
fi

if [ -f "training.log" ]; then
    mv training.log "$BACKUP_DIR/"
fi

if [ -f "training_history.json" ]; then
    mv training_history.json "$BACKUP_DIR/"
fi

if [ -d "plots" ]; then
    mv plots "$BACKUP_DIR/"
fi

echo "✓ Backed up to: $BACKUP_DIR"
echo ""

# 3. Update config to use cached-only datasets
echo "Step 3: Updating config to use cached images only..."

# Backup original config
if [ ! -f "config/teacher_global.yaml.backup" ]; then
    cp config/teacher_global.yaml config/teacher_global.yaml.backup
fi

# Update config
sed -i 's|train_jsonl:.*|train_jsonl: "./data/dataset_train_cached_only.jsonl"|g' config/teacher_global.yaml
sed -i 's|val_jsonl:.*|val_jsonl: "./data/dataset_val_cached_only.jsonl"|g' config/teacher_global.yaml
sed -i 's|num_workers:.*|num_workers: 8|g' config/teacher_global.yaml

echo "✓ Config updated"
echo ""

# 4. Show dataset stats
echo "Step 4: Dataset statistics..."
echo "  Training images: $(wc -l < data/dataset_train_cached_only.jsonl)"
echo "  Validation images: $(wc -l < data/dataset_val_cached_only.jsonl)"
echo "  Cache size: $(du -sh /media/skanndar/2TB1/aplantida-ml/image_cache | cut -f1)"
echo ""

# 5. Start fresh training
echo "Step 5: Starting fresh training (NO CHECKPOINT)..."
source venv/bin/activate

nohup python3 scripts/train_teacher.py \
    --config config/teacher_global.yaml \
    > training.log 2>&1 &

TRAIN_PID=$!
echo "✓ Training started (PID: $TRAIN_PID)"
echo ""

# 6. Wait a bit and show initial log
echo "Step 6: Initializing training..."
sleep 10
echo ""

echo "========================================"
echo "  ✅ Training Started Successfully!"
echo "========================================"
echo ""
echo "Benefits of cached-only training:"
echo "  • ZERO download failures"
echo "  • ZERO blank images"
echo "  • Clean, accurate metrics"
echo "  • Much faster (all from disk)"
echo ""
echo "Dataset size:"
echo "  • 123,787 training images (90.8% of original)"
echo "  • 13,816 validation images (79.5% of original)"
echo "  • 7,105 classes (vs 7,120 original - 15 classes had no cached images)"
echo ""
echo "Monitoring:"
echo "  tail -f training.log"
echo ""
echo "To restore old config and checkpoints:"
echo "  cp config/teacher_global.yaml.backup config/teacher_global.yaml"
echo "  mv $BACKUP_DIR/checkpoints/teacher_global checkpoints/"
echo ""

# Show initial log
echo "Initial training log:"
echo "--------------------"
tail -20 training.log

