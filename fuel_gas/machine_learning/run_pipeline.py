"""
End-to-End Execution Pipeline for Fuel Gas Soft Sensor (46-FIC-103A1.A):
1. Multi-model training, chronological validation, and production artifact export.
2. Full-historian batch inference and EWMA drift monitoring to DuckDB.
3. Evaluation figures and markdown engineering summary generation.
"""

import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.evaluate import run_evaluation
from src.inference import run_inference
from src.train import train


def main() -> None:
    t0 = time.time()
    print("=" * 75)
    print("      EXECUTING FUEL GAS SOFT SENSOR ML PIPELINE (46-FIC-103A1.A)      ")
    print("=" * 75)

    train()
    run_inference()
    run_evaluation()

    print(f"\n[DONE] Full ML pipeline finished in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()

