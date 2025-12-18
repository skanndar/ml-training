# Storage Migration to Secondary Disk

**Date**: 2025-12-18
**Reason**: Primary disk (nvme0) was 92% full, secondary disk (nvme1) has 441GB free

## What Was Moved

All heavy files were moved from the primary SSD to the secondary 2TB disk to free up space:

### 1. Image Cache (62GB)
**From**: `./data/image_cache`
**To**: `/media/skanndar/2TB1/aplantida-ml/image_cache`
**Symlink**: `data/image_cache -> /media/skanndar/2TB1/aplantida-ml/image_cache`

This is the LRU cache of downloaded plant images. Can grow up to 100GB.

### 2. Model Checkpoints (8.2GB)
**From**: `./checkpoints`
**To**: `/media/skanndar/2TB1/aplantida-ml/checkpoints`
**Symlink**: `checkpoints -> /media/skanndar/2TB1/aplantida-ml/checkpoints`

Contains all training checkpoints (epochs 1-15). Each checkpoint is ~350MB-1.1GB.

### 3. Training Results (16KB)
**From**: `./results`
**To**: `/media/skanndar/2TB1/aplantida-ml/results`
**Symlink**: `results -> /media/skanndar/2TB1/aplantida-ml/results`

Training history and metrics.

## Disk Usage After Migration

**Primary Disk (nvme0n1p2)**:
- Before: 405GB / 468GB (92% full) ⚠️
- After: 335GB / 468GB (76% full) ✅
- **Freed**: ~70GB

**Secondary Disk (nvme1n1)**:
- Before: 1.3TB / 1.8TB (75% full)
- After: 1.4TB / 1.8TB (77% full)
- **Used**: ~70GB

## Directory Structure

```
/home/skanndar/SynologyDrive/local/aplantida/ml-training/
├── data/
│   ├── image_cache -> /media/skanndar/2TB1/aplantida-ml/image_cache (symlink)
│   ├── dataset_train.jsonl
│   ├── dataset_val.jsonl
│   └── dataset_test.jsonl
├── checkpoints -> /media/skanndar/2TB1/aplantida-ml/checkpoints (symlink)
├── results -> /media/skanndar/2TB1/aplantida-ml/results (symlink)
└── ...

/media/skanndar/2TB1/aplantida-ml/
├── image_cache/          (62GB - LRU cache of plant images)
├── checkpoints/          (8.2GB - training checkpoints)
│   └── teacher_global/
│       ├── checkpoint_epoch_*.pt
│       ├── best_model.pt
│       └── last_checkpoint.pt
└── results/              (16KB - training metrics)
    └── teacher_global_v1/
        └── training_history.json
```

## How It Works

The migration uses **symbolic links (symlinks)** which act as transparent redirects. The training script continues to access files at the original paths (e.g., `./data/image_cache`) but the actual data is stored on the 2TB disk.

**Advantages**:
- ✅ No code changes needed
- ✅ Training continues seamlessly
- ✅ Easy to verify (symlinks visible with `ls -la`)
- ✅ Can be reverted if needed

## Training Impact

**Before Migration**:
- Training paused with `pkill -STOP`
- Files moved with `mv`
- Symlinks created with `ln -s`
- Training resumed with `pkill -CONT`

**Downtime**: ~30 seconds

**After Migration**:
- Training resumed normally
- Cache reads/writes now go to 2TB disk
- Checkpoints save to 2TB disk
- No performance impact (both are NVMe SSDs)

## Monitoring

Updated monitor script shows storage on both disks:

```bash
./monitor_training.sh
```

Output includes:
```
Storage Status:
-------------------------------------------------------------------
Cache size (2TB disk): 62G

/dev/nvme0n1p2 - Used: 335G / 468G (76% full)
/dev/nvme1n1 - Used: 1.4T / 1.8T (77% full)
```

## Future Growth Estimates

**Cache** (image_cache):
- Current: 62GB
- Max configured: 100GB
- Expected final: ~80-90GB (depending on dataset size)

**Checkpoints** (per training run):
- Teacher Global: ~8GB (15 epochs × 350MB-1.1GB)
- Teacher Regional: ~3GB (20 epochs × 150MB)
- Student: ~2GB (30 epochs × 70MB)
- **Total estimated**: ~15GB for complete pipeline

**Results**:
- Negligible (few MB)

**Total Space Needed on 2TB**:
- Current: 70GB
- Final (all models): ~120GB
- **Available**: 401GB (plenty of space)

## Recovery

If you need to move files back to primary disk:

```bash
# Stop training
pkill -STOP -f train_teacher.py

# Remove symlinks
rm data/image_cache checkpoints results

# Move data back
mv /media/skanndar/2TB1/aplantida-ml/image_cache data/
mv /media/skanndar/2TB1/aplantida-ml/checkpoints .
mv /media/skanndar/2TB1/aplantida-ml/results .

# Resume training
pkill -CONT -f train_teacher.py
```

## Notes

- ✅ Both disks are NVMe SSDs (no performance difference)
- ✅ 2TB disk is persistent (mounted at boot)
- ✅ Training can access files transparently through symlinks
- ✅ TensorBoard logs still work
- ✅ Backup scripts will need to include `/media/skanndar/2TB1/aplantida-ml/`

## Verification

To verify everything is working:

```bash
# Check symlinks
ls -la data/image_cache checkpoints results

# Check actual files
ls -lh /media/skanndar/2TB1/aplantida-ml/checkpoints/teacher_global/

# Check training is running
pgrep -f train_teacher.py

# Monitor training
./monitor_training.sh
```

All checks should pass with training continuing normally.
