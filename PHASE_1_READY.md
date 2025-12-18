# Phase 1: Teacher Global Training - READY TO START ✅

**Status**: Implementation complete, tested, ready for full training
**Date**: 2025-12-18
**Estimated Time**: 15-20 hours GPU training (or 40+ hours CPU)

## Overview

Phase 1 implements training for the first teacher model (ViT-Base) on the full global dataset. The model will be trained on 136,312 images from 7,120 plant species.

## Implementation Complete

### Training Script
**File**: [scripts/train_teacher.py](scripts/train_teacher.py)

**Features**:
- ✅ Mixed precision training (FP16) - reduces memory usage by ~50%
- ✅ Gradient accumulation support
- ✅ Learning rate warmup + cosine decay
- ✅ Top-1/Top-5 accuracy metrics
- ✅ Checkpoint saving/resuming
- ✅ TensorBoard logging
- ✅ Debug mode for testing
- ✅ Tested successfully on CPU

**Metrics Tracked**:
- Training/validation loss
- Top-1 accuracy (exact match)
- Top-5 accuracy (correct class in top 5 predictions)
- Learning rate per step
- Gradient norms (optional)

### Configuration
**File**: [config/teacher_global.yaml](config/teacher_global.yaml)

**Model**:
- Architecture: ViT-Base (Vision Transformer)
- Parameters: 91,273,936 (91M)
- Input size: 224x224
- Pre-trained: ImageNet-21k → ImageNet-1k
- Source: HuggingFace `timm` library

**Training Hyperparameters**:
- Batch size: 64
- Epochs: 15
- Learning rate: 1e-5 (with 2-epoch warmup)
- Optimizer: AdamW (weight_decay=0.01)
- Label smoothing: 0.1
- Gradient clipping: 1.0

**Data Augmentation**:
- Training:
  - Resize to 256x256
  - Random crop to 224x224
  - Horizontal flip (p=0.5)
  - Rotation (±15°, p=0.3)
  - Color jitter (brightness/contrast/saturation/hue)
  - ImageNet normalization

- Validation:
  - Resize to 256x256
  - Center crop to 224x224
  - ImageNet normalization (no augmentations)

### Dataset Statistics

**Training Set**:
- Images: 136,312 unique images
- Classes: 7,120 plant species
- Avg samples/class: 19.1
- License compliance: 92.4% permissive ✅

**Validation Set**:
- Images: 17,372
- Classes: 4,147 (subset of training classes)
- Used for early stopping and best model selection

**Streaming Approach**:
- Images downloaded on-the-fly during training
- LRU cache: 100GB max (configurable)
- First epoch slower (downloading), subsequent epochs fast (cached)
- No need to pre-download 136k images

## How to Run

### Option 1: Debug Mode (Quick Test)
Test on small subset to verify everything works:

```bash
cd /home/skanndar/SynologyDrive/local/aplantida/ml-training
source venv/bin/activate

# Test with 1000 samples (~5 minutes on CPU, ~2 minutes on GPU)
python scripts/train_teacher.py \
  --config ./config/teacher_global.yaml \
  --debug \
  --max-samples 1000
```

### Option 2: Full Training (CPU - NOT RECOMMENDED)
**Warning**: CPU training will be VERY slow (40-60 hours for 15 epochs)

```bash
python scripts/train_teacher.py \
  --config ./config/teacher_global.yaml
```

### Option 3: Full Training (GPU - RECOMMENDED)
**Requirements**: NVIDIA GPU with CUDA, 8GB+ VRAM

```bash
# Install GPU-enabled PyTorch first
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Run training
python scripts/train_teacher.py \
  --config ./config/teacher_global.yaml
```

**Estimated Time** (NVIDIA RTX 3090 or similar):
- Time per epoch: ~1 hour
- Total training (15 epochs): ~15 hours
- Total with validation: ~16-18 hours

### Option 4: Resume from Checkpoint
If training is interrupted, resume from last checkpoint:

```bash
python scripts/train_teacher.py \
  --config ./config/teacher_global.yaml \
  --resume ./checkpoints/teacher_global/last_checkpoint.pt
```

## Outputs

### Checkpoints
Saved in `checkpoints/teacher_global/`:

- `best_model.pt` - Best model based on validation Top-1 accuracy
- `last_checkpoint.pt` - Latest epoch (for resuming)
- `checkpoint_epoch_N.pt` - Saved every epoch
- `logs/` - TensorBoard logs

**Checkpoint contents**:
```python
{
  'epoch': int,
  'global_step': int,
  'model_state_dict': OrderedDict,  # Model weights
  'optimizer_state_dict': dict,     # Optimizer state
  'scheduler_state_dict': dict,     # LR scheduler state
  'metrics': dict,                  # Train/val metrics
  'config': dict,                   # Full training config
  'best_val_metric': float          # Best validation accuracy
}
```

### Results
Saved in `results/teacher_global_v1/`:

- `training_history.json` - Training curves (loss, accuracy per epoch)
- Future: `confusion_matrix.png`, `calibration_plot.png`

### TensorBoard Visualization
Monitor training in real-time:

```bash
# In separate terminal
tensorboard --logdir ./checkpoints/teacher_global/logs --port 6006

# Open browser: http://localhost:6006
```

**What you'll see**:
- Training/validation loss curves
- Top-1/Top-5 accuracy curves
- Learning rate schedule
- Gradient norms (if enabled)

## Expected Performance

Based on similar ViT-Base models trained on plant datasets:

**After 15 epochs**:
- Train Top-1: 60-75%
- Train Top-5: 80-90%
- Val Top-1: 45-60% (expected to be lower due to many classes)
- Val Top-5: 70-85%

**Why validation accuracy is lower**:
- 7,120 classes is a LOT (ImageNet has 1,000)
- Many species look very similar
- Training set has unbalanced distribution (some classes have 2 samples, others 100+)
- This is expected and will improve with:
  - Longer training (20-25 epochs)
  - More data per class
  - Ensemble with regional teacher

## Monitoring Training

### What to Watch For

**Good signs** ✅:
- Training loss steadily decreasing
- Validation accuracy increasing
- Top-5 accuracy >> Top-1 (shows model is learning but needs more training)
- Learning rate decreasing smoothly after warmup

**Warning signs** ⚠️:
- Validation loss increasing while training loss decreases (overfitting)
  - Solution: Reduce epochs, increase regularization
- Loss stuck at high value (~9.0 for 7120 classes)
  - Solution: Check learning rate, verify data loading
- GPU out of memory
  - Solution: Reduce batch size (64 → 32 → 16)
- Images not downloading (stuck at 0%)
  - Solution: Check internet connection, verify MongoDB is running

### Memory Requirements

**GPU (Mixed Precision FP16)**:
- Model: ~350MB
- Batch (64 images): ~2GB
- Gradients + optimizer: ~1.5GB
- **Total**: ~4GB VRAM minimum (recommend 8GB+)

**CPU (FP32)**:
- Model: ~700MB
- Batch: ~4GB
- **Total**: ~8GB RAM minimum (recommend 16GB+)

**Disk (LRU Cache)**:
- Cache limit: 100GB (configurable in config)
- After first epoch: ~40-60GB cached
- Subsequent epochs: minimal new downloads

## Troubleshooting

### Issue: GPU Out of Memory
```bash
# Reduce batch size
nano config/teacher_global.yaml
# Change: batch_size: 64 → batch_size: 32

# Or enable gradient accumulation (2 steps = effective batch 128)
# Add to config:
training:
  gradient_accumulation_steps: 2
```

### Issue: Training Too Slow on CPU
```bash
# Reduce number of workers
nano config/teacher_global.yaml
# Change: num_workers: 4 → num_workers: 2

# Or reduce batch size
batch_size: 64 → batch_size: 16
```

### Issue: Images Not Downloading
```bash
# Check MongoDB is running
docker ps | grep mongo

# Test streaming dataset
python scripts/test_streaming.py --sample-size 20

# Check internet connection
ping google.com
```

### Issue: Loss is NaN
```bash
# Reduce learning rate
nano config/teacher_global.yaml
# Change: learning_rate: 1.0e-5 → learning_rate: 5.0e-6

# Enable gradient clipping (already enabled)
gradient_clip: 1.0
```

## After Training Completes

Once training finishes successfully:

1. **Evaluate best model**:
```bash
python scripts/evaluate_model.py \
  --checkpoint ./checkpoints/teacher_global/best_model.pt \
  --test-jsonl ./data/dataset_test.jsonl
```

2. **Review TensorBoard**:
   - Check if training converged
   - Look for overfitting (val loss increasing)
   - Verify accuracy plateaued

3. **Proceed to Phase 2**: Train regional teacher on EU_SW subset
   - Can initialize from global teacher checkpoint
   - Only 24k images, faster training (~8-10 hours)

## Next Steps

After Phase 1 completes:

**Phase 2: Teacher Regional (EU_SW)**
- Dataset: 24,666 images, 728 species
- Initialize from: `checkpoints/teacher_global/best_model.pt` (optional)
- Fine-tune on European flora
- Expected improvement: +15% accuracy on EU species

**Phase 3: Student Distillation (MobileNetV2)**
- Distill knowledge from both teachers
- Model size: 3.5M params (26x smaller than ViT-Base)
- Target: 80% of teacher accuracy with 10x faster inference

**Phase 4: TensorFlow.js Export**
- Convert to web format
- Quantize to FP16 (~7MB model)
- Deploy to PWA with Service Worker caching

## Summary

Phase 1 is **READY TO START**:

✅ Training script implemented and tested
✅ Configuration files ready
✅ Dataset prepared (136k training, 17k validation)
✅ Streaming pipeline working
✅ Checkpoint saving/resuming
✅ TensorBoard logging
✅ All dependencies installed

**To start training**:
```bash
cd /home/skanndar/SynologyDrive/local/aplantida/ml-training
source venv/bin/activate

# Option 1: Quick test (5 min)
python scripts/train_teacher.py --config ./config/teacher_global.yaml --debug --max-samples 1000

# Option 2: Full training (15-20 hours on GPU)
python scripts/train_teacher.py --config ./config/teacher_global.yaml
```

**Current Status**: Waiting for user decision to start full training or continue with next phases.

---

**Notes**:
- User is still importing data (currently 340k images, target 900k)
- Can start training now with current data
- When full dataset ready, re-train for better accuracy
- Training can be interrupted and resumed anytime

**Recommendation**: Start debug mode test now to verify GPU/CUDA setup, then start full training if everything works.
