#!/usr/bin/env python3
"""
Real-time training monitor - Shows progress from TensorBoard logs
"""

import time
import os
from pathlib import Path
from datetime import datetime, timedelta
import json
import glob

def get_latest_tfevent_data():
    """Extract info from latest TensorBoard event file."""
    logs_dir = Path("/media/skanndar/2TB1/aplantida-ml/checkpoints/teacher_global/logs")

    if not logs_dir.exists():
        return None

    # Find latest event file
    events = glob.glob(str(logs_dir / "events.out.tfevents*"))
    if not events:
        return None

    latest_event = max(events, key=os.path.getmtime)
    return latest_event

def get_checkpoints_info():
    """Get info about checkpoints."""
    checkpoint_dir = Path("/media/skanndar/2TB1/aplantida-ml/checkpoints/teacher_global")

    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    epochs_completed = len(checkpoints)

    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        latest_epoch = int(latest_checkpoint.name.split("_")[-1].split(".")[0])
        return epochs_completed, latest_epoch

    return 0, 0

def get_training_history():
    """Get training history if it exists."""
    history_file = Path("/media/skanndar/2TB1/aplantida-ml/results/teacher_global_v1/training_history.json")

    if history_file.exists() and history_file.stat().st_size > 0:
        try:
            with open(history_file, 'r') as f:
                return json.load(f)
        except:
            return None

    return None

def main():
    print("\n" + "="*70)
    print("  APLANTIDA TRAINING REAL-TIME MONITOR")
    print("="*70)
    print()

    start_time = datetime.now()

    while True:
        try:
            # Get status
            epochs_completed, latest_epoch = get_checkpoints_info()
            history = get_training_history()

            elapsed = datetime.now() - start_time

            # Display header
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] ", end="", flush=True)
            print(f"Elapsed: {elapsed.total_seconds()/3600:.1f}h | ", end="", flush=True)
            print(f"Epochs: {epochs_completed}/15", end="", flush=True)

            # Show latest metrics if available
            if history:
                if len(history.get('train_loss', [])) > 0:
                    latest_loss = history['train_loss'][-1]
                    latest_top1 = history['train_top1_acc'][-1]
                    print(f" | Loss: {latest_loss:.4f} | Top-1: {latest_top1:.2f}%", end="", flush=True)

            time.sleep(5)

        except KeyboardInterrupt:
            print("\n\n✓ Monitor stopped")
            break
        except Exception as e:
            print(f"\n⚠️  Error: {e}")
            time.sleep(10)

if __name__ == '__main__':
    main()
