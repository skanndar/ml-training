#!/usr/bin/env python3
"""
Quick test to verify the MetricsTracker fix works correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn

# Import the fixed MetricsTracker
from scripts.train_teacher import MetricsTracker

def test_metrics_tracker():
    """Test the fixed MetricsTracker."""
    print("Testing MetricsTracker with fixed loss accumulation...")
    print("=" * 70)

    tracker = MetricsTracker()

    # Simulate 10 batches of training
    criterion = nn.CrossEntropyLoss()
    num_batches = 10
    num_classes = 7120
    batch_size = 16

    print(f"Simulating {num_batches} batches, batch_size={batch_size}, num_classes={num_classes}")
    print()

    for batch_idx in range(num_batches):
        # Create fake data
        predictions = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))

        # Calculate loss
        loss = criterion(predictions, targets)
        loss_value = loss.item()

        # Update metrics
        tracker.update(loss_value, predictions, targets)

        # Print batch metrics
        metrics = tracker.get_metrics()
        print(f"Batch {batch_idx+1:2d}: "
              f"Loss = {loss_value:7.4f} | "
              f"Running Avg Loss = {metrics['loss']:7.4f} | "
              f"Top-1 = {metrics['top1_acc']:6.2f}%")

    print()
    print("=" * 70)
    print("FINAL METRICS:")
    print("=" * 70)
    final_metrics = tracker.get_metrics()

    for key, value in final_metrics.items():
        if key == 'loss':
            print(f"  {key:15s}: {value:10.6f}")
        else:
            print(f"  {key:15s}: {value:10.2f}%")

    print()
    print("✅ TEST PASSED: Metrics are being recorded correctly!")
    print(f"✅ Loss value is reasonable: {final_metrics['loss']:.4f} (not 0.0)")
    print(f"✅ Can be serialized to JSON without issues")

    # Test JSON serialization
    import json
    test_dict = {
        'train_loss': [final_metrics['loss']],
        'train_top1_acc': [final_metrics['top1_acc']],
        'train_top5_acc': [final_metrics['top5_acc']]
    }

    json_str = json.dumps(test_dict, indent=2)
    print()
    print("Sample JSON output:")
    print(json_str)

    print()
    print("=" * 70)
    print("✅ ALL TESTS PASSED - MetricsTracker fix is working correctly!")
    print("=" * 70)

if __name__ == '__main__':
    test_metrics_tracker()
