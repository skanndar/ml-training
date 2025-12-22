#!/bin/bash
# Quick reference: commands cheat sheet
# Copiar → pegar en terminal

# ============================================================
# 0. SETUP (Only once)
# ============================================================

# Create directories
mkdir -p ~/ml-training/{data,checkpoints,results,config,scripts}
cd ~/ml-training

# Clone documentation
cp -r /home/skanndar/SynologyDrive/local/rehabProyectos/docs/aplantida-ml-training/* ./

# Setup Python
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate (Windows)
pip install -r requirements.txt

# ============================================================
# 1. PHASE 0: AUDIT (First day)
# ============================================================

# Check dataset
python scripts/audit_dataset.py \
  --log_level INFO

# Validate URLs (sample 1000)
python scripts/validate_image_urls.py

# Export from Mongo
python scripts/export_dataset.py \
  --output_dir ./data \
  --log_level INFO

# Create train/val/test splits
python scripts/split_dataset.py \
  --jsonl_path ./data/dataset_raw.jsonl \
  --output_dir ./data \
  --seed 42

# ============================================================
# 2. PHASE 1: TEACHER GLOBAL (2-3 days)
# ============================================================

# Train
python scripts/train_teacher.py \
  --config config/teacher_global.yaml \
  --output_dir ./results/teacher_global_v1 \
  --seed 42

# Evaluate
python scripts/eval_teacher.py \
  --model ./results/teacher_global_v1/best_model.pt \
  --test_jsonl ./data/dataset_splits.jsonl \
  --output ./results/teacher_global_v1/eval_test.json

# Compute logits (for distillation)
python scripts/compute_teacher_logits.py \
  --model ./results/teacher_global_v1/best_model.pt \
  --train_jsonl ./data/dataset_splits.jsonl \
  --output ./data/teacher_global_logits_train.npz

# ============================================================
# 3. PHASE 2: TEACHER REGIONAL (1-2 days)
# ============================================================

# Prepare regional subset
python scripts/prepare_regional_dataset.py \
  --input_jsonl ./data/dataset_splits.jsonl \
  --output_dir ./data \
  --region EU_SW

# Train regional teacher
python scripts/train_teacher.py \
  --config config/teacher_regional.yaml \
  --output_dir ./results/teacher_regional_v1 \
  --seed 42

# Evaluate by region
python scripts/eval_by_region.py \
  --model ./results/teacher_regional_v1/best_model.pt \
  --test_jsonl ./data/dataset_splits.jsonl \
  --output ./results/teacher_regional_v1/eval_by_region.json

# Compute regional logits
python scripts/compute_teacher_logits.py \
  --model ./results/teacher_regional_v1/best_model.pt \
  --train_jsonl ./data/dataset_splits.jsonl \
  --output ./data/teacher_regional_logits_train.npz

# ============================================================
# 4. PHASE 3: PREPARE DISTILLATION (1 day)
# ============================================================

# Combine soft labels from multiple teachers
python scripts/combine_teacher_logits.py \
  --teacher_global ./data/teacher_global_logits_train.npz \
  --teacher_regional ./data/teacher_regional_logits_train.npz \
  --weights 0.5 0.5 \
  --output ./data/soft_labels_combined_train.npz

# ============================================================
# 5. PHASE 4: TRAIN STUDENT (3-5 days)
# ============================================================

# Distillation phase (10 epochs)
python scripts/train_student_distill.py \
  --config config/student.yaml \
  --soft_labels ./data/soft_labels_combined_train.npz \
  --output_dir ./results/student_distill_v1 \
  --seed 42

# Fine-tuning phase (10 epochs)
python scripts/train_student_finetune.py \
  --config config/student.yaml \
  --checkpoint ./results/student_distill_v1/best_model.pt \
  --output_dir ./results/student_finetune_v1 \
  --seed 42

# ============================================================
# 6. PHASE 5: EVALUATION & CALIBRATION (1 day)
# ============================================================

# Full evaluation
python scripts/full_evaluation.py \
  --model ./results/student_finetune_v1/best_model.pt \
  --test_jsonl ./data/dataset_splits.jsonl \
  --output ./results/student_finetune_v1/full_evaluation.json

# Calibrate model
python scripts/calibrate_model.py \
  --model ./results/student_finetune_v1/best_model.pt \
  --val_jsonl ./data/dataset_splits.jsonl \
  --output ./results/student_finetune_v1/calibration.json

# Find optimal confidence threshold
python scripts/find_confidence_threshold.py \
  --model ./results/student_finetune_v1/best_model.pt \
  --val_jsonl ./data/dataset_splits.jsonl \
  --output ./results/student_finetune_v1/threshold.json

# ============================================================
# 7. PHASE 6: EXPORT TO TF.JS (1-2 days)
# ============================================================

# Quantize to FP16
python scripts/quantize_model.py \
  --model ./results/student_finetune_v1/best_model.pt \
  --quantization_type fp16 \
  --output ./results/student_finetune_v1/model_fp16.pt

# Export to SavedModel
python scripts/export_to_savedmodel.py \
  --model ./results/student_finetune_v1/model_fp16.pt \
  --output_dir ./results/student_finetune_v1

# Convert to TF.js
python scripts/export_to_tfjs.py \
  --model ./results/student_finetune_v1/model_fp16.pt \
  --output_dir ./dist/models/student_v1.0 \
  --format tfjs \
  --quantization fp16

# Validate TF.js export
python scripts/validate_tfjs_export.py \
  --original_model ./results/student_finetune_v1/model_fp16.pt \
  --tfjs_model ./dist/models/student_v1.0

# ============================================================
# 8. PHASE 7: REPRODUCIBILITY (1 day)
# ============================================================

# Generate manifest
python scripts/generate_manifest.py \
  --phase_results ./results \
  --output ./results/TRAINING_MANIFEST_v1.0.yaml

# Check legal compliance
python scripts/license_checker.py \
  --manifest ./results/TRAINING_MANIFEST_v1.0.yaml

# Create visualization dashboard
python scripts/dashboard.py \
  --evaluation_json ./results/student_finetune_v1/full_evaluation.json \
  --output ./results/eval_dashboard.png

# ============================================================
# 9. DEPLOY TO PWA (Frontend)
# ============================================================

# 1. Copy model to frontend
cp -r ./dist/models/student_v1.0 ../aplantidaFront/public/models/

# 2. Update PlantRecognition.js
# → See EXPORT_TFJS_PWA.md § 4.1

# 3. Register Service Worker
# → See EXPORT_TFJS_PWA.md § 4.2

# 4. Build frontend
cd ../aplantidaFront
npm run build

# 5. Test offline
# → Start dev server, disconnect network, test recognition

# ============================================================
# MONITORING & DEBUGGING
# ============================================================

# Tensorboard (real-time training)
tensorboard --logdir ./results --port 6006
# Then open: http://localhost:6006

# Check logs
tail -f ./logs/training.log

# Test on sample
python scripts/test_inference.py \
  --model ./results/student_finetune_v1/best_model.pt \
  --test_image ./data/test_sample.jpg

# Compare vs teachers
python scripts/compare_student_vs_ensemble.py \
  --student ./results/student_finetune_v1/best_model.pt \
  --teacher_global ./results/teacher_global_v1/best_model.pt \
  --teacher_regional ./results/teacher_regional_v1/best_model.pt \
  --test_set ./data/dataset_splits.jsonl

# ============================================================
# ONE-LINER: FULL PIPELINE
# ============================================================

# If you want everything at once:
./scripts/reproduce_full_pipeline.sh

# Or manually:
# bash -c 'python scripts/audit_dataset.py && \
#          python scripts/export_dataset.py && \
#          python scripts/train_teacher.py --config config/teacher_global.yaml && \
#          ... etc'

# ============================================================
# USEFUL REFERENCES
# ============================================================

# Quick evaluation snapshot
python -c "
import json
with open('./results/student_finetune_v1/full_evaluation.json') as f:
    data = json.load(f)
    print(f'Top-1: {data[\"global\"][\"top1\"]:.4f}')
    print(f'Top-5: {data[\"global\"][\"top5\"]:.4f}')
    print(f'ECE: {data[\"ece\"]:.4f}')
"

# Check model size
du -h ./results/student_finetune_v1/best_model.pt
du -h ./dist/models/student_v1.0/

# Count samples by split
python -c "
import pandas as pd
df = pd.read_json('./data/dataset_splits.jsonl', lines=True)
print(df['split'].value_counts())
"

# ============================================================
# CLEANUP (If restarting)
# ============================================================

# Remove training artifacts
rm -rf ./checkpoints/*
rm -rf ./results/*
rm -rf ./data/soft_labels*
rm -rf ./data/teacher*logits*

# Keep: data cache, raw data
# Rerun: from Phase 1

# ============================================================
# STATUS CHECK
# ============================================================

echo "✓ Setup complete? $(ls venv 2>/dev/null && echo 'YES' || echo 'NO')"
echo "✓ Data ready? $([ -f data/dataset_splits.jsonl ] && echo 'YES' || echo 'NO')"
echo "✓ Models trained? $([ -d results ] && echo 'YES' || echo 'NO')"
echo "✓ TF.js exported? $([ -f dist/models/student_v1.0/model.json ] && echo 'YES' || echo 'NO')"

