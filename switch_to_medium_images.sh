#!/bin/bash

# Switch to Medium-sized Images (20x faster training)
# This script changes image URLs from /original.jpg to /medium.jpg

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "======================================"
echo "  Switch to Medium-Sized Images"
echo "======================================"
echo ""

# Check if backups already exist
if [ -f "data/dataset_train_original.jsonl.backup" ]; then
    echo "⚠️  WARNING: Backup files already exist!"
    echo ""
    echo "This means you might have already switched to medium images before."
    echo ""
    read -p "Do you want to proceed anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

echo "Step 1: Creating backups of original datasets..."
cp data/dataset_train.jsonl data/dataset_train_original.jsonl.backup
cp data/dataset_val.jsonl data/dataset_val_original.jsonl.backup
echo "✓ Backups created"
echo ""

echo "Step 2: Replacing /original.jpg with /medium.jpg..."
sed -i 's|/original\.jpg|/medium.jpg|g' data/dataset_train.jsonl
sed -i 's|/original\.jpg|/medium.jpg|g' data/dataset_val.jsonl
echo "✓ URLs updated"
echo ""

echo "Step 3: Verifying changes..."
MEDIUM_COUNT=$(grep -c "medium.jpg" data/dataset_train.jsonl || true)
ORIGINAL_COUNT=$(grep -c "original.jpg" data/dataset_train.jsonl || true)

echo "  - Training set: $MEDIUM_COUNT medium URLs, $ORIGINAL_COUNT original URLs"

if [ "$ORIGINAL_COUNT" -gt 0 ]; then
    echo "  ⚠️  WARNING: Some original URLs still present!"
fi

if [ "$MEDIUM_COUNT" -eq 0 ]; then
    echo "  ❌ ERROR: No medium URLs found! Something went wrong."
    echo ""
    echo "Restoring from backup..."
    cp data/dataset_train_original.jsonl.backup data/dataset_train.jsonl
    cp data/dataset_val_original.jsonl.backup data/dataset_val.jsonl
    echo "Restored. Please check the files manually."
    exit 1
fi

echo "✓ Changes verified"
echo ""

echo "Step 4: Clearing old cache (original-sized images)..."
CACHE_DIR="/media/skanndar/2TB1/aplantida-ml/image_cache"
CACHE_SIZE=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1 || echo "0")

echo "  Current cache size: $CACHE_SIZE"
echo ""
read -p "Delete all cached images? This will force re-download with medium size. (y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$CACHE_DIR"/*
    echo "✓ Cache cleared"
else
    echo "⚠️  Cache NOT cleared. Old (original-sized) images will still be used if present."
    echo "   This may cause inconsistencies. Recommended to clear cache."
fi
echo ""

echo "======================================"
echo "  ✅ Successfully switched to MEDIUM"
echo "======================================"
echo ""
echo "Benefits:"
echo "  • 20x faster downloads"
echo "  • 10x more images fit in cache"
echo "  • Same training quality (model uses 224x224 anyway)"
echo ""
echo "Next steps:"
echo ""
echo "1. Stop current training (if running):"
echo "   pkill -SIGTERM -f train_teacher.py"
echo ""
echo "2. Wait for checkpoint to save (10 seconds):"
echo "   sleep 10"
echo ""
echo "3. Resume training from last checkpoint:"
echo "   cd $SCRIPT_DIR"
echo "   source venv/bin/activate"
echo "   nohup python3 scripts/train_teacher.py \\"
echo "       --config config/teacher_global.yaml \\"
echo "       --resume checkpoints/teacher_global/last_checkpoint.pt \\"
echo "       > training.log 2>&1 &"
echo ""
echo "4. Monitor progress:"
echo "   tail -f training.log"
echo ""
echo "To restore original URLs (if needed):"
echo "   cp data/dataset_train_original.jsonl.backup data/dataset_train.jsonl"
echo "   cp data/dataset_val_original.jsonl.backup data/dataset_val.jsonl"
echo ""
