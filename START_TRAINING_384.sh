#!/bin/bash
# Start training with 384px + Smart Crop + Stratified Split
# Version: Teacher Global v2

set -e  # Exit on error

echo "========================================================================"
echo "  Starting Training: 384px + Smart Crop + Stratified Split"
echo "========================================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "config/teacher_global.yaml" ]; then
    echo "❌ Error: Run this script from ml-training directory"
    exit 1
fi

# Activate venv
source venv/bin/activate

echo "✅ Virtual environment activated"
echo ""

# Check stratified datasets exist
if [ ! -f "data/dataset_train_stratified.jsonl" ]; then
    echo "⚠️  Stratified datasets not found. Creating them..."
    python3 scripts/create_stratified_split.py
    echo ""
fi

echo "✅ Stratified datasets ready"
echo ""

# Backup old checkpoints if they exist
if [ -d "checkpoints/teacher_global" ]; then
    backup_dir="checkpoints/teacher_global_224_backup_$(date +%Y%m%d_%H%M%S)"
    echo "⚠️  Found existing checkpoints (incompatible with 384px)"
    echo "   Backing up to: $backup_dir"
    mv checkpoints/teacher_global "$backup_dir"
    echo "   ✅ Backup complete"
    echo ""
fi

# Show config summary
echo "========================================================================"
echo "  Configuration Summary"
echo "========================================================================"
echo "Model: vit_base_patch16_384 (90M parameters)"
echo "Image size: 384x384 pixels"
echo "Batch size: 8 (reduced from 16 due to larger images)"
echo "Smart crop: Enabled (saliency-based)"
echo "Dataset: Stratified split"
echo "  - Train: data/dataset_train_stratified.jsonl"
echo "  - Val:   data/dataset_val_stratified.jsonl"
echo ""
echo "Expected improvements:"
echo "  - 3x more detail (384 vs 224 pixels)"
echo "  - Smart crop focused on plants"
echo "  - Proper validation (all classes in both train/val)"
echo ""
echo "Trade-offs:"
echo "  - ~60% slower training (70-80 min vs 45 min per epoch)"
echo "  - Higher VRAM usage (~3.8GB vs 3.2GB)"
echo ""

# Ask for confirmation
read -p "Start training? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Training cancelled"
    exit 0
fi

echo ""
echo "========================================================================"
echo "  Starting Training..."
echo "========================================================================"
echo ""

# Start training in background
nohup python3 scripts/train_teacher.py \
    --config config/teacher_global.yaml \
    > training_384_smartcrop.log 2>&1 &

TRAIN_PID=$!

echo "✅ Training started!"
echo ""
echo "Process ID: $TRAIN_PID"
echo "Log file: training_384_smartcrop.log"
echo ""
echo "Monitoring commands:"
echo "  - View progress:  tail -f training_384_smartcrop.log | grep -E 'Epoch|loss|top1'"
echo "  - Full log:       tail -f training_384_smartcrop.log"
echo "  - Stop training:  kill $TRAIN_PID"
echo "  - GPU usage:      nvidia-smi -l 2"
echo ""
echo "Expected completion: ~18-20 hours (15 epochs × 75 min/epoch)"
echo ""

# Show initial logs
echo "========================================================================"
echo "  Initial Training Output (first 30 seconds)"
echo "========================================================================"
echo ""

sleep 30
tail -50 training_384_smartcrop.log

echo ""
echo "========================================================================"
echo "  Training is running in background"
echo "========================================================================"
echo ""
echo "Monitor with: tail -f training_384_smartcrop.log | grep -E 'Epoch|loss|top1'"
echo ""
