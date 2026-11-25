# Fuel Gas Machine Learning Soft Sensor (`46-FIC-103A1.A`)

Standalone Machine Learning module for predicting Stage-1 Fuel Gas flowrate **`46-FIC-103A1.A` [LBS/HR]** using **7 primary process features** (`46-TIC-101.A`, `46-TI-102.A`, `46-PIC-109.A`, `46-TIC-106`, `46-FIC-101_PV`, `46-AI-101.A`, `46-FIC-103A1.Q`) from the InfluxDB measurement `historian_clean`, with explicit splits across **8 training intervals** (`../data/fuel_valid_train_data.csv`) and **6 blind test intervals** (`../data/fuel_valid_test_data.csv`).

---

## Directory Structure

```text
machine_learning/
├── README.md
├── __init__.py
├── run_pipeline.py                      # End-to-end runner (train -> inference -> evaluate)
├── src/
│   ├── __init__.py
│   ├── config.py                        # InfluxDB config, train/test CSVs, & 7 feature tags
│   ├── features.py                      # 7 process features & deterministic physics gating
│   ├── models.py                        # 4 candidate models (StackedEnsemble [GroupKFold], RandomForest, XGBoost, MLP)
│   ├── train.py                         # GroupKFold selection & blind evaluation on fuel_valid_test_data.csv
│   ├── inference.py                     # FuelGasSoftSensorEngine & CLI (--single-point, --latest, etc.)
│   └── evaluate.py                      # 300 DPI evaluation figures & TRAINING_REPORT.md from InfluxDB
└── models/
    ├── softsensor_103a1.joblib          # Trained best model + feature metadata
    └── training_evaluation_summary.json
```

---

## Execution & Inference

```bash
# Run full ML pipeline (train -> inference -> evaluate)
python machine_learning/run_pipeline.py

# Run individual steps
python machine_learning/src/train.py
python machine_learning/src/inference.py                  # Full historian -> InfluxDB
python machine_learning/src/inference.py --single-point   # 1 latest point (EWMA disabled)
python machine_learning/src/evaluate.py                   # Generate figures & report in reports/
```

```python
from machine_learning import FuelGasSoftSensorEngine

engine = FuelGasSoftSensorEngine()
# res = engine.predict_single(row_dict)      # Real-time SCADA/DCS step
# df_out = engine.predict_dataframe(df_raw)  # Batch DataFrame
```
