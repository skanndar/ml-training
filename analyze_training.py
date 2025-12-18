#!/usr/bin/env python3
"""
analyze_training.py - Visualize and analyze training results

This script creates a comprehensive analysis of training metrics without requiring TensorFlow.
"""

import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib not installed. Install with: pip install matplotlib")


def load_history(history_file):
    """Load training history from JSON."""
    if not Path(history_file).exists():
        print(f"❌ File not found: {history_file}")
        return None

    with open(history_file, 'r') as f:
        return json.load(f)


def print_metrics_table(history):
    """Print training metrics in a nice table."""
    print("\n" + "="*100)
    print(" TRAINING METRICS - EPOCH BY EPOCH")
    print("="*100)
    print(f"{'Epoch':<6} {'Train Loss':<14} {'Train Top-1':<14} {'Train Top-5':<14} {'Val Loss':<14} {'Val Top-1':<14} {'Val Top-5':<14}")
    print("-"*100)

    train_loss = history.get('train_loss', [])
    train_top1 = history.get('train_top1_acc', [])
    train_top5 = history.get('train_top5_acc', [])
    val_loss = history.get('val_loss', [])
    val_top1 = history.get('val_top1_acc', [])
    val_top5 = history.get('val_top5_acc', [])

    num_epochs = max(
        len(train_loss), len(train_top1), len(val_loss),
        len(val_top1), len(train_top5), len(val_top5)
    )

    for epoch in range(num_epochs):
        epoch_num = epoch + 1
        tl = train_loss[epoch] if epoch < len(train_loss) else 0
        tt1 = train_top1[epoch] if epoch < len(train_top1) else 0
        tt5 = train_top5[epoch] if epoch < len(train_top5) else 0
        vl = val_loss[epoch] if epoch < len(val_loss) else 0
        vt1 = val_top1[epoch] if epoch < len(val_top1) else 0
        vt5 = val_top5[epoch] if epoch < len(val_top5) else 0

        print(
            f"{epoch_num:<6} "
            f"{tl:<14.4f} "
            f"{tt1:<14.2f}% "
            f"{tt5:<14.2f}% "
            f"{vl:<14.4f} "
            f"{vt1:<14.2f}% "
            f"{vt5:<14.2f}%"
        )

    print("="*100)


def print_summary_stats(history):
    """Print summary statistics."""
    print("\n" + "="*100)
    print(" SUMMARY STATISTICS")
    print("="*100)

    train_loss = np.array(history.get('train_loss', []))
    train_top1 = np.array(history.get('train_top1_acc', []))
    val_loss = np.array(history.get('val_loss', []))
    val_top1 = np.array(history.get('val_top1_acc', []))

    # Filter out zeros for meaningful statistics
    train_loss_valid = train_loss[train_loss > 0]
    train_top1_valid = train_top1[train_top1 > 0]
    val_top1_valid = val_top1[val_top1 > 0]

    print("\n📊 TRAINING LOSS:")
    if len(train_loss_valid) > 0:
        print(f"  Initial:     {train_loss_valid[0]:.4f}")
        print(f"  Final:       {train_loss_valid[-1]:.4f}")
        print(f"  Best:        {train_loss_valid.min():.4f}")
        print(f"  Avg:         {train_loss_valid.mean():.4f}")
        print(f"  Improvement: {(train_loss_valid[0] - train_loss_valid[-1]):.4f} ({(train_loss_valid[0] - train_loss_valid[-1])/train_loss_valid[0]*100:.1f}%)")
    else:
        print(f"  All values are 0 or invalid")

    print("\n📊 TRAINING TOP-1 ACCURACY:")
    if len(train_top1_valid) > 0:
        print(f"  Initial:     {train_top1_valid[0]:.2f}%")
        print(f"  Final:       {train_top1_valid[-1]:.2f}%")
        print(f"  Best:        {train_top1_valid.max():.2f}%")
        print(f"  Avg:         {train_top1_valid.mean():.2f}%")
        print(f"  Improvement: {(train_top1_valid[-1] - train_top1_valid[0]):.2f}%")
    else:
        print(f"  All values are 0 or invalid")

    print("\n📊 VALIDATION LOSS:")
    if len(val_loss) > 0:
        print(f"  Initial:     {val_loss[0]:.4f}")
        print(f"  Final:       {val_loss[-1]:.4f}")
        print(f"  Best:        {val_loss.min():.4f}")
        print(f"  Avg:         {val_loss.mean():.4f}")
    else:
        print(f"  All values are 0 or invalid")

    print("\n📊 VALIDATION TOP-1 ACCURACY:")
    if len(val_top1_valid) > 0:
        print(f"  Initial:     {val_top1_valid[0]:.2f}%")
        print(f"  Final:       {val_top1_valid[-1]:.2f}%")
        print(f"  Best:        {val_top1_valid.max():.2f}%")
        print(f"  Avg:         {val_top1_valid.mean():.2f}%")
    else:
        print(f"  All values are 0 or invalid")

    print("\n" + "="*100)


def create_plots(history, output_dir="results/"):
    """Create visualization plots."""
    if not HAS_MATPLOTLIB:
        print("⚠️  Cannot create plots without matplotlib")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    epochs = list(range(1, len(history.get('train_loss', [])) + 1))

    # Plot 1: Loss Curves
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs, history.get('train_loss', []), 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax.plot(epochs, history.get('val_loss', []), 'r-s', label='Val Loss', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=150)
    print(f"✅ Saved: {output_dir / 'loss_curves.png'}")
    plt.close()

    # Plot 2: Top-1 Accuracy
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs, history.get('train_top1_acc', []), 'b-o', label='Train Top-1', linewidth=2, markersize=4)
    ax.plot(epochs, history.get('val_top1_acc', []), 'r-s', label='Val Top-1', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Top-1 Accuracy (Exact Match)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'top1_accuracy.png', dpi=150)
    print(f"✅ Saved: {output_dir / 'top1_accuracy.png'}")
    plt.close()

    # Plot 3: Top-5 Accuracy
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs, history.get('train_top5_acc', []), 'g-o', label='Train Top-5', linewidth=2, markersize=4)
    ax.plot(epochs, history.get('val_top5_acc', []), 'orange', marker='s', label='Val Top-5', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Top-5 Accuracy (Top 5 Predictions)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'top5_accuracy.png', dpi=150)
    print(f"✅ Saved: {output_dir / 'top5_accuracy.png'}")
    plt.close()

    # Plot 4: Combined metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    ax1.plot(epochs, history.get('train_loss', []), 'b-o', label='Train', linewidth=2, markersize=4)
    ax1.plot(epochs, history.get('val_loss', []), 'r-s', label='Val', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Top-1
    ax2.plot(epochs, history.get('train_top1_acc', []), 'b-o', label='Train', linewidth=2, markersize=4)
    ax2.plot(epochs, history.get('val_top1_acc', []), 'r-s', label='Val', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Top-1 Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Top-5
    ax3.plot(epochs, history.get('train_top5_acc', []), 'g-o', label='Train', linewidth=2, markersize=4)
    ax3.plot(epochs, history.get('val_top5_acc', []), 'orange', marker='s', label='Val', linewidth=2, markersize=4)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Top-5 Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Summary stats as text
    ax4.axis('off')
    train_loss_val = history.get('train_loss', [])
    val_loss_val = history.get('val_loss', [])
    train_top1_val = history.get('train_top1_acc', [])
    val_top1_val = history.get('val_top1_acc', [])

    summary_text = f"""
TRAINING SUMMARY

📊 Loss:
  Train Final: {train_loss_val[-1]:.4f}
  Val Final:   {val_loss_val[-1]:.4f}

🎯 Top-1 Accuracy:
  Train Final: {train_top1_val[-1]:.2f}%
  Val Final:   {val_top1_val[-1]:.2f}%

📈 Training Progress:
  Total Epochs: {len(epochs)}
  Loss Improvement: {(train_loss_val[0] - train_loss_val[-1]):.4f}
"""
    ax4.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'training_summary.png', dpi=150)
    print(f"✅ Saved: {output_dir / 'training_summary.png'}")
    plt.close()


def print_interpretation(history):
    """Print interpretation of the results."""
    print("\n" + "="*100)
    print(" 📖 CÓMO INTERPRETAR ESTOS RESULTADOS")
    print("="*100)

    print("""
🔍 LOSS (Pérdida/Error):
   - Mide qué tan lejos están las predicciones del modelo de la respuesta correcta
   - Valores BAJOS = Buen modelo ✅
   - DEBE DISMINUIR con cada época (curva hacia abajo)
   - Objetivo: Busca una curva decreciente y que se estabilice

🎯 TOP-1 ACCURACY (Exactitud del Top-1):
   - Porcentaje de veces que el modelo PREDICE CORRECTAMENTE en el primer intento
   - 0% = Adivina al azar
   - 100% = Predicción perfecta
   - En 7,120 clases, esperar ~2-5% es razonable (es muy difícil)

🔝 TOP-5 ACCURACY (Exactitud del Top-5):
   - Porcentaje de veces que la respuesta CORRECTA está entre las 5 predicciones principales
   - Mucho más alto que Top-1 (debería ser ~5-10x más alto)
   - Indica cuántas buenas predicciones hace el modelo

📈 TRAIN vs VALIDATION:
   - TRAIN: Métricas en los datos de entrenamiento
   - VAL: Métricas en datos nuevos que el modelo no ha visto
   - Si TRAIN >> VAL = OVERFITTING (el modelo solo memorizó) ⚠️
   - Si son similares = Buen generalization ✅

⚙️ LEARNING RATE (Velocidad de aprendizaje):
   - Comienza alta (aprendizaje rápido)
   - Disminuye progresivamente (ajuste fino)
   - Debe ver una curva suave que baja con el tiempo
""")

    # Analyze actual metrics
    print("\n" + "-"*100)
    print(" ✅ DIAGNÓSTICO DE TU ENTRENAMIENTO")
    print("-"*100)

    train_loss = np.array(history.get('train_loss', []))
    train_top1 = np.array(history.get('train_top1_acc', []))
    val_loss = np.array(history.get('val_loss', []))
    val_top1 = np.array(history.get('val_top1_acc', []))

    # Check for zero values
    if np.all(train_loss == 0) or np.all(train_top1 == 0):
        print("""
⚠️  PROBLEMA DETECTADO:
   Las métricas de TRAIN muestran valores cero (0.0)

   Esto significa que el modelo NO se está entrenando correctamente.

   Posibles causas:
   1. ❌ El training_loss no se está calculando
   2. ❌ Hay un problema con el backward pass
   3. ❌ Los datos no están cargándose correctamente
   4. ❌ Problema con TensorFlow/PyTorch

   SOLUCIÓN:
   Necesitamos revisar los logs de entrenamiento para encontrar el error.
""")
    else:
        print("""
✅ El modelo SE ESTÁ ENTRENANDO:
   - Train loss está bajando (buen aprendizaje)
   - Accuracy está mejorando
   - Sin signos de overfitting severo
""")

    if len(val_loss) > 0 and val_loss[0] > 0:
        print(f"\n   Validation Loss inicial: {val_loss[0]:.4f}")
        print(f"   Validation Loss final:   {val_loss[-1]:.4f}")
        if val_loss[-1] < val_loss[0]:
            print("   ✅ Validación mejorando con el tiempo")
        else:
            print("   ⚠️  Validación no mejora (posible overfitting)")

    print("\n" + "="*100)


def main():
    history_file = "results/teacher_global_v1/training_history.json"

    print("\n" + "="*100)
    print(" 🤖 APLANTIDA - TRAINING RESULTS ANALYZER")
    print("="*100)

    history = load_history(history_file)
    if not history:
        print("❌ No se pudo cargar el archivo de historial")
        return

    # Print metrics table
    print_metrics_table(history)

    # Print summary statistics
    print_summary_stats(history)

    # Print interpretation guide
    print_interpretation(history)

    # Create plots
    if HAS_MATPLOTLIB:
        print("\n📊 Generando gráficas...")
        create_plots(history, "results/teacher_global_v1/")
        print("\n✅ Gráficas guardadas en: results/teacher_global_v1/")
    else:
        print("\n⚠️  Instala matplotlib para ver gráficas: pip install matplotlib")


if __name__ == '__main__':
    main()
