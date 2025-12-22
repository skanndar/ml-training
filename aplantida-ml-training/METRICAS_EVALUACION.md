# Métricas y Evaluación

## Métricas Core

### 1.1 - Top-K Accuracy

```python
# scripts/metrics.py

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

def topk_accuracy(predictions, targets, k=(1, 5)):
    """
    Args:
        predictions: (N, 9000) logits del modelo
        targets: (N,) índices verdaderos

    Returns:
        dict con top-1, top-5
    """
    _, top_k_preds = torch.topk(predictions, k=max(k), dim=1)  # (N, 5)

    results = {}
    for ki in k:
        matches = (top_k_preds[:, :ki] == targets.unsqueeze(1)).any(dim=1)
        results[f"top{ki}"] = matches.float().mean().item()

    return results

# Uso:
# {
#   "top1": 0.72,
#   "top5": 0.88
# }
```

**Interpretación:**
- Top-1: ¿primera predicción es correcta? (lo que ve el usuario)
- Top-5: ¿está entre top-5? (tolerancia para ranking)

---

### 1.2 - Accuracy por Región

```python
def accuracy_by_region(predictions, targets, regions, region_names=None):
    """
    Evalúa accuracy por región geográfica.

    Args:
        predictions: (N, 9000)
        targets: (N,)
        regions: (N,) región para cada muestra ["EU_SW", "EU", "AMERICAS", ...]
        region_names: list de regiones a evaluar

    Returns:
        dict {region: {"top1": ..., "top5": ..., "samples": ...}}
    """
    region_names = region_names or set(regions)
    results = {}

    for region in region_names:
        mask = torch.tensor([r == region for r in regions])

        if mask.sum() == 0:
            continue

        preds_region = predictions[mask]
        targets_region = targets[mask]

        metrics = topk_accuracy(preds_region, targets_region)
        metrics["samples"] = mask.sum().item()
        results[region] = metrics

    return results

# Uso esperado:
# {
#   "EU_SW": {"top1": 0.82, "top5": 0.92, "samples": 15000},
#   "EU": {"top1": 0.70, "top5": 0.85, "samples": 12000},
#   "AMERICAS": {"top1": 0.65, "top5": 0.80, "samples": 8000},
#   "OTHER": {"top1": 0.68, "top5": 0.83, "samples": 50000}
# }

# Objetivo:
# - EU_SW +15% respecto global (teacher regional funciona)
# - Otras regiones ~3-5% por debajo global (acceptable)
```

---

### 1.3 - Per-Class Metrics

```python
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def per_class_metrics(predictions, targets, class_names=None):
    """
    Calcula precision, recall, F1 por clase.
    Útil para detectar clases problemáticas (imbalance).
    """
    preds = torch.argmax(predictions, dim=1)

    precision, recall, f1, support = precision_recall_fscore_support(
        targets.cpu().numpy(),
        preds.cpu().numpy(),
        average=None,
        zero_division=0
    )

    # Identificar clases problemáticas
    low_recall = [(i, recall[i]) for i in range(len(recall)) if recall[i] < 0.5]
    low_recall.sort(key=lambda x: x[1])

    print(f"Classes with recall < 0.5: {low_recall[:20]}")

    return {
        "precision": precision.mean(),
        "recall": recall.mean(),
        "f1": f1.mean(),
        "per_class_recall": recall,
        "per_class_precision": precision
    }
```

---

## Calibración

### 2.1 - Expected Calibration Error (ECE)

```python
def expected_calibration_error(predictions, targets, num_bins=15):
    """
    ECE: promedio de diferencia entre confianza predicha y accuracy real.

    Ejemplo:
    - Predicción con confianza 0.9 debería tener accuracy ~0.9
    - Si accuracy actual es 0.7 → descalibrado

    Args:
        predictions: (N, 9000) logits
        targets: (N,)
        num_bins: número de bins de confianza

    Returns:
        ece: float entre 0 y 1
    """
    # Confianza máxima por predicción
    confidences = torch.softmax(predictions, dim=1).max(dim=1)[0]

    # Predicciones
    preds = predictions.argmax(dim=1)
    correctness = (preds == targets).float()

    # Dividir en bins de confianza
    bins = torch.linspace(0, 1, num_bins + 1)
    ece = 0

    for i in range(num_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i+1])

        if mask.sum() == 0:
            continue

        avg_confidence = confidences[mask].mean()
        avg_accuracy = correctness[mask].mean()

        ece += mask.sum() / len(predictions) * abs(avg_confidence - avg_accuracy)

    return ece.item()

# Interpretación:
# ECE < 0.05: Excelente calibración
# ECE 0.05-0.10: Buena
# ECE 0.10-0.20: Mediocre (aplicar temperature scaling)
# ECE > 0.20: Mal calibrado
```

### 2.2 - Temperature Scaling

```python
def find_optimal_temperature(predictions, targets, val_range=(0.5, 5.0)):
    """
    Encuentra temperatura que minimiza ECE.
    """
    temperatures = torch.linspace(val_range[0], val_range[1], 50)
    eces = []

    for T in temperatures:
        scaled_preds = predictions / T
        ece = expected_calibration_error(scaled_preds, targets)
        eces.append(ece)

    best_temp = temperatures[torch.tensor(eces).argmin()].item()
    best_ece = min(eces)

    print(f"Optimal temperature: {best_temp:.3f}")
    print(f"Best ECE: {best_ece:.4f}")

    return best_temp

# Aplicar temperatura óptima en inference:
logits = model(image)
scaled_logits = logits / optimal_temperature  # T ~1.2-1.5 típico
probs = softmax(scaled_logits)
```

---

## Matriz de Confusión

### 3.1 - Top-20 Confusiones

```python
from sklearn.metrics import confusion_matrix

def analyze_confusions(predictions, targets, class_names, top_k=20):
    """
    Identifica las 20 confusiones más comunes.
    Útil para detectar clases similares.
    """
    preds = predictions.argmax(dim=1)

    # Confusiones donde predicted != target
    confused = preds != targets
    confused_preds = preds[confused]
    confused_targets = targets[confused]

    # Contar pares (true_class, predicted_class)
    confusion_pairs = {}
    for true_cls, pred_cls in zip(confused_targets, confused_preds):
        pair = (true_cls.item(), pred_cls.item())
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

    # Top-20
    top_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:20]

    print("Top-20 confusions (true_class → predicted_class):")
    for (true_cls, pred_cls), count in top_confusions:
        true_name = class_names.get(true_cls, f"cls_{true_cls}")
        pred_name = class_names.get(pred_cls, f"cls_{pred_cls}")
        print(f"  {true_name} → {pred_name}: {count} times")

    return top_confusions
```

**Acción:** Si una especie confunde frecuentemente:
1. Aumentar sus muestras en training
2. Añadir augmentations específicas
3. Re-entrenar modelo

---

## Umbral "No Conclusive"

### 4.1 - Análisis de confianza

```python
def analyze_confidence_threshold(predictions, targets):
    """
    Encuentra threshold óptimo para "no conclusive".
    """
    confidences = torch.softmax(predictions, dim=1).max(dim=1)[0]
    preds = predictions.argmax(dim=1)
    correctness = (preds == targets).float()

    # Para cada threshold, calcular coverage y precision
    thresholds = torch.linspace(0.3, 0.95, 50)
    results = []

    for threshold in thresholds:
        mask = confidences >= threshold

        if mask.sum() == 0:
            continue

        coverage = mask.sum() / len(predictions)  # % de predicciones aceptadas
        precision = correctness[mask].mean()       # accuracy en samples aceptadas

        results.append({
            "threshold": threshold.item(),
            "coverage": coverage.item(),
            "precision": precision.item(),
            "accuracy": (precision * coverage).item()
        })

    # Encontrar threshold para 95% precision
    for r in results:
        if r["precision"] >= 0.95:
            print(f"Threshold {r['threshold']:.3f}: "
                  f"precision={r['precision']:.4f}, coverage={r['coverage']:.4f}")
            break

    return results

# Típicamente:
# - threshold=0.70 → 85-90% precision, 90% coverage
# - threshold=0.80 → 95% precision, 70% coverage
# - threshold=0.85 → 98% precision, 50% coverage

# RECOMENDACIÓN:
# Usar threshold=0.75 para balance entre coverage y precision
```

### 4.2 - Usar threshold en app

```javascript
// PlantRecognition.js (frontend)

async function recognizeWithThreshold(imageElement, threshold = 0.75) {
  const logits = await model.predict(preprocessed_image);
  const probs = tf.softmax(logits);

  // Top-1
  const top1_prob = tf.max(probs).dataSync()[0];

  if (top1_prob < threshold) {
    return {
      success: false,
      message: `No conclusive (confidence ${top1_prob.toFixed(2)} < ${threshold})`,
      results: null
    };
  }

  // Mostrar resultado
  const top5 = tf.topk(probs, 5);
  return {
    success: true,
    results: mapToSpecies(top5)
  };
}
```

---

## Full Evaluation Pipeline

### 5.1 - Script completo

```python
# scripts/full_evaluation.py

import torch
import json
from pathlib import Path

def run_full_evaluation(model_path: str, test_loader, device="cuda"):
    """
    Ejecuta todas las métricas en un dataset.
    """
    model = load_model(model_path)
    model.eval()
    model = model.to(device)

    all_predictions = []
    all_targets = []
    all_regions = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            targets = batch["hard_label"].to(device)
            regions = batch["region"]

            logits = model(images)
            all_predictions.append(logits)
            all_targets.append(targets)
            all_regions.extend(regions)

    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Computar todas las métricas
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "num_samples": len(targets),

        # Accuracy global
        "global": topk_accuracy(predictions, targets),

        # Por región
        "by_region": accuracy_by_region(predictions, targets, all_regions),

        # Calibración
        "ece": expected_calibration_error(predictions, targets),
        "optimal_temperature": find_optimal_temperature(predictions, targets),

        # Per-class
        "per_class": per_class_metrics(predictions, targets),

        # Confianza
        "confidence_analysis": analyze_confidence_threshold(predictions, targets)
    }

    # Guardar
    output_file = Path(model_path).parent / "full_evaluation.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation saved to {output_file}")
    print(json.dumps(results, indent=2))

    return results
```

---

## Criterios de Éxito Finales

```yaml
evaluation_success_criteria:
  global:
    top1_accuracy: >= 0.75
    top5_accuracy: >= 0.90

  regional:
    eu_sw_top1: >= 0.88        # +15% respecto global
    eu_top1: >= 0.68           # -3% respecto global
    americas_top1: >= 0.65
    other_top1: >= 0.70

  per_class:
    avg_recall: >= 0.70        # Incluso clases imbalanceadas
    min_recall: >= 0.40        # Clases raras

  calibration:
    ece: <= 0.10               # Expected Calibration Error
    mce: <= 0.25               # Max Calibration Error

  confidence:
    precision_at_95_conf: >= 0.95
    coverage_at_95_conf: >= 0.70

  confusion:
    top_confusion_rate: <= 5%  # % de muestras en top confusión
```

---

## Dashboard de Monitoreo (Opcional)

```python
# scripts/dashboard.py - Visibilidad en tiempo real

import matplotlib.pyplot as plt
import seaborn as sns

def plot_evaluation_dashboard(results_json: str):
    """
    Crea dashboard visual de todas las métricas.
    """
    with open(results_json) as f:
        results = json.load(f)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Global accuracy
    ax = axes[0, 0]
    ax.bar(["Top-1", "Top-5"], [results["global"]["top1"], results["global"]["top5"]])
    ax.set_ylim([0, 1])
    ax.set_title("Global Accuracy")

    # 2. By region
    ax = axes[0, 1]
    regions = list(results["by_region"].keys())
    top1s = [results["by_region"][r]["top1"] for r in regions]
    ax.bar(regions, top1s)
    ax.set_ylim([0, 1])
    ax.set_title("Top-1 by Region")
    ax.tick_params(axis='x', rotation=45)

    # 3. Calibration
    ax = axes[0, 2]
    ax.text(0.5, 0.7, f"ECE: {results['ece']:.4f}", ha='center', fontsize=20)
    ax.text(0.5, 0.3, f"Temp: {results['optimal_temperature']:.3f}", ha='center', fontsize=20)
    ax.axis('off')

    # 4. Confidence distribution
    ax = axes[1, 0]
    confs = results["confidence_analysis"]
    thresholds = [c["threshold"] for c in confs]
    precisions = [c["precision"] for c in confs]
    ax.plot(thresholds, precisions, marker='o')
    ax.axhline(y=0.95, color='r', linestyle='--', label='95% target')
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Precision")
    ax.set_title("Confidence vs Precision")
    ax.legend()
    ax.grid()

    # 5. Per-class metrics
    ax = axes[1, 1]
    ax.text(0.5, 0.7, f"Avg Recall: {results['per_class']['recall']:.4f}", ha='center', fontsize=16)
    ax.text(0.5, 0.4, f"Avg F1: {results['per_class']['f1']:.4f}", ha='center', fontsize=16)
    ax.axis('off')

    # 6. Samples per region
    ax = axes[1, 2]
    regions = list(results["by_region"].keys())
    samples = [results["by_region"][r]["samples"] for r in regions]
    ax.pie(samples, labels=regions, autopct='%1.1f%%')
    ax.set_title("Dataset Distribution")

    plt.tight_layout()
    plt.savefig("evaluation_dashboard.png", dpi=300, bbox_inches='tight')
    print("Saved dashboard to evaluation_dashboard.png")
```

---

## Resumen Checklist de Evaluación

- [ ] Global Top-1 >= 75%
- [ ] Global Top-5 >= 90%
- [ ] Spain/SW Europe +15% vs global
- [ ] Per-class recall > 60% (incluso imbalanceadas)
- [ ] ECE < 0.10
- [ ] Confidence threshold @95% precision identified
- [ ] Confusion matrix top-20 analizado
- [ ] Dashboard generado
