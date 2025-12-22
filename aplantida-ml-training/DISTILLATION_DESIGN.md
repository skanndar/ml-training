# Distillation Design: Arquitectura Exacta

## Concepto

Knowledge distillation: un modelo pequeño (student) aprende de modelos grandes (teachers) sin tener que cargarlos todos al desplegar.

**Nuestro caso:**
- 3 Teachers (global + regional + opcional) ≈ 450MB cada uno
- 1 Student (MobileNetV2) ≈ 15MB
- Resultado: rendimiento ≈ ensemble, tamaño ≈ 1/30

---

## Arquitectura Student

### Configuración exacta

```python
# models/student_architecture.py

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

class StudentPlantRecognition(nn.Module):
    def __init__(self, num_classes: int = 9000, dropout_rate: float = 0.2):
        super().__init__()

        # Base: MobileNetV2 pretrained
        self.backbone = mobilenet_v2(pretrained=True)

        # Reemplazar clasificador
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def get_intermediate_features(self, x):
        """Para análisis de distillation."""
        return self.backbone.features(x)
```

**Por qué MobileNetV2:**
- Diseñado para eficiencia (inverted residuals)
- Buen balance accuracy/latencia
- Nativo en TF.js
- ~15MB base, ~5-7MB cuantizado

---

## Distillation Loss

### 1. Loss Function

```python
# models/distillation_loss.py

import torch
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature: float = 3.0, alpha: float = 0.7):
        """
        Args:
            temperature: Controla suavidad de soft targets (3.0 recomendado)
            alpha: Balance entre KL divergence (0.7) y cross-entropy (0.3)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, hard_labels):
        """
        Args:
            student_logits: (batch, 9000) logits del student
            teacher_logits: (batch, 9000) logits promedio teachers
            hard_labels: (batch,) índices de clases verdaderas

        Returns:
            loss: escalar
        """

        # Soft targets (teachers softened)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)

        # Student softmax
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)

        # KL Divergence (distillation loss)
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean")
        kl_loss *= self.temperature ** 2  # Rescaling necesario

        # Cross-entropy con hard labels
        ce_loss = F.cross_entropy(student_logits, hard_labels)

        # Combinación
        total_loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss

        return total_loss
```

### 2. Temperatura

**Qué es:** parámetro que controla la "suavidad" de las predicciones.

- `T = 1.0` → predicciones sharp (típico)
- `T = 3.0` → predicciones smooth (mejor para distillation)
- `T = 10.0` → muy smooth (pierde información)

**Intuición:**
```
teacher logits:  [10, 8, 2]  (claro: clase 0)
T=1:  softmax -> [0.98, 0.02, 0.0]  (sharp)
T=3:  softmax -> [0.70, 0.25, 0.05]  (suave, útil para student)
```

**Valor recomendado:** T = 3.0 (ajustable experimentalmente)

---

## Combinación de Multiple Teachers

### Estrategia: Weighted Ensemble Logits

```python
# models/teacher_ensemble.py

class TeacherEnsemble:
    def __init__(self, teacher_paths: list, weights: list, region_filters: list = None):
        """
        Args:
            teacher_paths: ["path/teacher_global.pt", "path/teacher_regional.pt"]
            weights: [0.5, 0.5]  # deben sumar 1
            region_filters: [None, "EU_SW"]  # filtrar regional por región
        """
        self.teachers = [torch.load(p) for p in teacher_paths]
        self.weights = weights
        self.region_filters = region_filters or [None] * len(teacher_paths)

        for teacher in self.teachers:
            teacher.eval()

    def get_ensemble_logits(self, images, regions=None):
        """
        Args:
            images: (batch, 3, 224, 224)
            regions: (batch,) región para cada imagen (ej. ["EU_SW", "EU", ...])

        Returns:
            ensemble_logits: (batch, 9000) promedio ponderado
        """
        ensemble_logits = None

        for teacher, weight, region_filter in zip(self.teachers, self.weights, self.region_filters):
            teacher_logits = teacher(images)  # (batch, 9000)

            # Si hay filtro regional, usar ponderación condicional
            if region_filter and regions is not None:
                mask = torch.tensor([r == region_filter for r in regions])
                # Solo aplicar este teacher donde region == region_filter
                # (Versión simple: usar siempre)
                pass

            if ensemble_logits is None:
                ensemble_logits = weight * teacher_logits
            else:
                ensemble_logits += weight * teacher_logits

        return ensemble_logits
```

### Alternativa: Gating Network (avanzada)

Si quieres que el ensemble sea inteligente (no solo promedio):

```python
class GatingTeacherEnsemble(nn.Module):
    def __init__(self, num_teachers: int = 3, num_classes: int = 9000):
        super().__init__()

        # Red que aprende pesos dinámicos
        self.gating_network = nn.Sequential(
            nn.Linear(num_classes * num_teachers, 512),
            nn.ReLU(),
            nn.Linear(512, num_teachers),
            nn.Softmax(dim=1)
        )

    def forward(self, teacher_logits_list):
        """
        teacher_logits_list: list de (batch, 9000) tensores
        """
        concatenated = torch.cat(teacher_logits_list, dim=1)  # (batch, 27000)
        weights = self.gating_network(concatenated)  # (batch, 3)

        # Aplicar pesos
        ensemble = torch.zeros_like(teacher_logits_list[0])
        for i, logits in enumerate(teacher_logits_list):
            ensemble += weights[:, i:i+1] * logits

        return ensemble
```

**Recomendación:** Empezar con weighted average (simple), pasar a gating solo si resultados no convergen.

---

## Generación de Soft Labels

### Precomputed (recomendado)

```python
# scripts/generate_soft_labels.py

import torch
import numpy as np
from tqdm import tqdm

def generate_soft_labels_precomputed(
    train_loader,
    teacher_ensemble,
    output_file: str,
    temperature: float = 3.0,
    device: str = "cuda"
):
    """
    Precomputa logits de teachers en todo el train set.
    """
    all_logits = []
    all_indices = []

    teacher_ensemble.eval()

    with torch.no_grad():
        for batch_idx, (images, indices) in enumerate(tqdm(train_loader)):
            images = images.to(device)

            # Obtener logits ensemble
            logits = teacher_ensemble.get_ensemble_logits(images)
            all_logits.append(logits.cpu().numpy())
            all_indices.extend(indices.cpu().numpy())

            if (batch_idx + 1) % 100 == 0:
                print(f"Processed {batch_idx+1} batches")

    # Concatenar
    all_logits = np.concatenate(all_logits, axis=0)  # (850k, 9000)
    all_indices = np.array(all_indices)

    # Guardar
    np.savez_compressed(
        output_file,
        logits=all_logits,
        indices=all_indices
    )

    print(f"Saved soft labels to {output_file}")
    print(f"Shape: {all_logits.shape}")

# Uso:
teacher_ensemble = TeacherEnsemble(...)
train_loader = get_train_dataloader(...)
generate_soft_labels_precomputed(
    train_loader, teacher_ensemble,
    output_file="./data/soft_labels_train.npz",
    temperature=3.0
)
```

### On-the-fly (si memoria limitada)

```python
# En train_student.py

class OnTheFlyDistillationDataset(torch.utils.data.Dataset):
    def __init__(self, train_loader, teacher_ensemble, temperature=3.0):
        self.train_loader = train_loader
        self.teacher_ensemble = teacher_ensemble
        self.temperature = temperature
        self.teacher_ensemble.eval()

    def __getitem__(self, idx):
        image, hard_label, index = self.train_loader.dataset[idx]

        # Generar soft label on-the-fly
        with torch.no_grad():
            image_tensor = transform(image).unsqueeze(0).to(device)
            teacher_logits = self.teacher_ensemble.get_ensemble_logits(image_tensor)
            soft_probs = torch.softmax(teacher_logits / self.temperature, dim=1)

        return {
            "image": image_tensor,
            "hard_label": hard_label,
            "soft_target": soft_probs.squeeze(0)
        }
```

**Trade-offs:**
- **Precomputed:** Rápido (no genera en cada epoch), requiere 3.3GB disco
- **On-the-fly:** Flexible, lento (3x más tiempo epoch)

---

## Training Loop Student

### Combinado: Distillation + Fine-tuning

```python
# scripts/train_student_combined.py

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

class StudentTrainer:
    def __init__(
        self,
        student_model: nn.Module,
        teacher_ensemble: TeacherEnsemble,
        train_loader,
        val_loader,
        device: str = "cuda",
        temperature: float = 3.0,
        alpha: float = 0.7,
        learning_rate: float = 1e-4
    ):
        self.student = student_model.to(device)
        self.teacher_ensemble = teacher_ensemble
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.temperature = temperature
        self.alpha = alpha

        self.optimizer = Adam(self.student.parameters(), lr=learning_rate)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=20,  # total epochs
            eta_min=1e-6
        )

        self.distillation_loss_fn = DistillationLoss(temperature, alpha)
        self.ce_loss_fn = torch.nn.CrossEntropyLoss()

    def train_epoch_distillation(self, epoch: int):
        """Fase 1: Distillation pura (epochs 0-10)."""
        self.student.train()
        self.teacher_ensemble.eval()

        total_loss = 0
        batch_count = 0

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            hard_labels = batch["hard_label"].to(self.device)

            # Logits ensemble (precomputed o on-the-fly)
            if "soft_target" in batch:
                # Precomputed
                soft_probs = batch["soft_target"].to(self.device)
                teacher_logits = torch.log(soft_probs + 1e-7)  # Convert back
            else:
                # On-the-fly
                with torch.no_grad():
                    teacher_logits = self.teacher_ensemble.get_ensemble_logits(images)

            # Student forward
            student_logits = self.student(images)

            # Loss: KL + CE (con weights)
            loss = self.distillation_loss_fn(
                student_logits, teacher_logits, hard_labels
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / batch_count
                print(f"Epoch {epoch} Batch {batch_idx+1}: Loss {avg_loss:.4f}")

        return total_loss / batch_count

    def train_epoch_finetuning(self, epoch: int):
        """Fase 2: Fine-tuning con hard labels (epochs 10-20)."""
        self.student.train()

        total_loss = 0
        batch_count = 0

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            hard_labels = batch["hard_label"].to(self.device)

            # Solo CE loss, sin teacher
            student_logits = self.student(images)
            loss = self.ce_loss_fn(student_logits, hard_labels)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        return total_loss / batch_count

    def validate(self, epoch: int):
        """Validación."""
        self.student.eval()

        total_loss = 0
        total_acc = 0
        batch_count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                hard_labels = batch["hard_label"].to(self.device)

                logits = self.student(images)
                loss = self.ce_loss_fn(logits, hard_labels)

                _, preds = torch.max(logits, 1)
                acc = (preds == hard_labels).float().mean()

                total_loss += loss.item()
                total_acc += acc.item()
                batch_count += 1

        avg_loss = total_loss / batch_count
        avg_acc = total_acc / batch_count

        print(f"Epoch {epoch} Val: Loss {avg_loss:.4f}, Acc {avg_acc:.4f}")
        return avg_loss, avg_acc

    def train(self, num_epochs_distill: int = 10, num_epochs_finetune: int = 10):
        """
        Entrenamiento completo:
        1. Distillation (epochs 0-10)
        2. Fine-tuning (epochs 10-20)
        """
        best_val_acc = 0

        # Fase 1: Distillation
        for epoch in range(num_epochs_distill):
            train_loss = self.train_epoch_distillation(epoch)
            val_loss, val_acc = self.validate(epoch)
            self.scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.student.state_dict(), f"./checkpoints/student_epoch_{epoch}.pt")

        # Fase 2: Fine-tuning
        for epoch in range(num_epochs_distill, num_epochs_distill + num_epochs_finetune):
            train_loss = self.train_epoch_finetuning(epoch)
            val_loss, val_acc = self.validate(epoch)
            self.scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.student.state_dict(), f"./checkpoints/student_epoch_{epoch}.pt")

        print(f"Best validation accuracy: {best_val_acc:.4f}")
```

---

## Validación: Comparar Student vs Ensemble

```python
# scripts/compare_student_vs_ensemble.py

def evaluate_student_vs_ensemble(
    student_model,
    teacher_ensemble,
    test_loader,
    device: str = "cuda"
):
    """
    Compara accuracy student vs ensemble en test set.
    """
    student_model.eval()
    teacher_ensemble.eval()

    student_correct = 0
    ensemble_correct = 0
    total = 0

    kl_divergences = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            hard_labels = batch["hard_label"].to(device)

            # Student prediction
            student_logits = student_model(images)
            student_probs = F.softmax(student_logits, dim=1)
            student_preds = torch.argmax(student_logits, dim=1)

            # Ensemble prediction
            ensemble_logits = teacher_ensemble.get_ensemble_logits(images)
            ensemble_probs = F.softmax(ensemble_logits, dim=1)
            ensemble_preds = torch.argmax(ensemble_logits, dim=1)

            # Accuracy
            student_correct += (student_preds == hard_labels).sum().item()
            ensemble_correct += (ensemble_preds == hard_labels).sum().item()
            total += hard_labels.size(0)

            # KL divergence
            kl = F.kl_div(
                F.log_softmax(student_logits, dim=1),
                F.softmax(ensemble_logits, dim=1),
                reduction="batchmean"
            )
            kl_divergences.append(kl.item())

    student_acc = student_correct / total
    ensemble_acc = ensemble_correct / total
    avg_kl = np.mean(kl_divergences)

    print(f"Student accuracy: {student_acc:.4f}")
    print(f"Ensemble accuracy: {ensemble_acc:.4f}")
    print(f"Accuracy gap: {ensemble_acc - student_acc:.4f}")
    print(f"Average KL divergence: {avg_kl:.4f}")

    return {
        "student_acc": student_acc,
        "ensemble_acc": ensemble_acc,
        "accuracy_gap": ensemble_acc - student_acc,
        "avg_kl": avg_kl
    }

# Criterio de éxito: accuracy gap < 0.05 (5%)
```

---

## Resumen: Parámetros Finales

```yaml
distillation:
  temperature: 3.0           # Suavidad soft labels
  alpha: 0.7                 # Peso KL vs CE
  teacher_weights: [0.5, 0.5]  # Promedio ensemble

student:
  architecture: mobilenetv2_1.0
  num_classes: 9000
  dropout: 0.2

training:
  phase1_epochs: 10          # Distillation pura
  phase2_epochs: 10          # Fine-tuning
  learning_rate: 1e-4
  batch_size: 128

success_criteria:
  student_top1: >= 0.70      # Cercano a teachers
  student_top5: >= 0.87
  accuracy_gap: < 0.05       # vs ensemble
  kl_divergence: < 0.2
```
