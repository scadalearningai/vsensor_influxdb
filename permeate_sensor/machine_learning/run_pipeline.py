"""
Execution Pipeline for Decoupled Soft Sensor (46-FQI-106A):
1. Model Training (LightGBM Decoupled 9-Features)
2. Batch Inference & Sensor Drift Evaluation (InfluxDB Export)
3. Comprehensive Performance Evaluation (Matplotlib Figures)
"""

import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.train import train
from src.inference import run_inference
from src.evaluate import run_evaluation

def main():
    start_time = time.time()
    print("=" * 70)
    print("      EXECUTING DECOUPLED SOFTSENSOR PIPELINE (46-FQI-106A)          ")
    print("=" * 70)

    print("\n>>> STEP 1: MODEL TRAINING & VALIDATION")
    train()

    print("\n>>> STEP 2: BATCH INFERENCE & DRIFT DETECTION (INFLUXDB EXPORT)")
    run_inference()

    print("\n>>> STEP 3: PERFORMANCE EVALUATION & FIGURE GENERATION")
    run_evaluation()

    elapsed = time.time() - start_time
    print(f"\n[DONE] Decoupled pipeline executed successfully in {elapsed:.1f} seconds.")

if __name__ == '__main__':
    main()
