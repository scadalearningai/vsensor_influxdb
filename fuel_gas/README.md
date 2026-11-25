# Fuel Gas Soft Sensor (`46-FIC-103A1.A`)

Virtual sensor / soft sensor module for estimating Stage-1 fuel gas flowrate (**`46-FIC-103A1.A`**) on Waste Heat Boiler #1 (WHB #1), Medco E&P Indonesia, connected to **InfluxDB v1.x**.

Each prediction approach is organized into a standalone submodule:

1. **[`interpolation/`](interpolation/README.md)** — **First-Principles Valve Characteristics** based on `data/cv_fuelgas.csv` (*Piecewise Linear Interpolation* for DCS/SCADA compatibility and *Natural Cubic Spline*), writing batch outputs to the InfluxDB measurement `predicted_fuel_gas_103a`.
2. **[`machine_learning/`](machine_learning/README.md)** — **Data-Driven Physical Soft Sensor** (`StackedEnsemble` with `GroupKFold` selected from 4 candidates: `StackedEnsemble`, `RandomForest`, `XGBoost`, `MLP`; using **7 stateless raw features**) trained on 8 intervals in `data/fuel_valid_train_data.csv` and blind-tested on 6 intervals in `data/fuel_valid_test_data.csv` from the InfluxDB measurement `historian_clean`.

---

## Directory Structure

```text
fuel_gas/
├── README.md
├── data/                                    # Valve curve table & train/test intervals
│   ├── cv_fuelgas.csv                       # Valve 46-FV-103A capacity curve (Opening vs Cv)
│   ├── fuel_valid_train_data.csv            # 8 valid operational windows for ML training
│   ├── fuel_valid_test_data.csv             # 6 blind test windows for ML evaluation
│   ├── hourly_prediction_2026-09-09_avg.csv
│   └── hourly_prediction_2026-09-09_exact.csv
├── interpolation/                           # [METHOD 1] Valve Characteristics Interpolation
│   ├── README.md
│   ├── __init__.py
│   ├── run_interpolation.py                 # Main CLI runner (linear & spline)
│   ├── src/
│   │   ├── __init__.py
│   │   ├── config.py                        # InfluxDB config, paths, & valve constants
│   │   ├── estimator.py                     # ValveFlowEstimator & InfluxDB pipeline
│   │   └── valve_spline.py                  # ValveSplineInterpolator (Cubic Spline)
│   └── scripts/
│       └── plot_prediction_results.py       # Plot generator from InfluxDB to reports/figures/
├── machine_learning/                        # [METHOD 2] Machine Learning Soft Sensor
│   ├── README.md
│   ├── __init__.py
│   ├── run_pipeline.py                      # End-to-end pipeline (train, inference, evaluate)
│   ├── src/
│   │   ├── __init__.py
│   │   ├── config.py                        # InfluxDB config, train/test CSVs, & 7 feature tags
│   │   ├── features.py                      # 7 raw features preparation & physics gating
│   │   ├── models.py                        # 4 candidates (StackedEnsemble, RF, XGBoost, MLP)
│   │   ├── train.py                         # Train GroupKFold & Blind Test evaluator
│   │   ├── inference.py                     # FuelGasSoftSensorEngine & CLI (--single-point, --latest, etc.)
│   │   └── evaluate.py                      # 300 DPI figures & TRAINING_REPORT.md
│   └── models/
│       ├── softsensor_103a1.joblib          # Selected production StackedEnsemble model
│       └── training_evaluation_summary.json
└── references/                              # Raw historian visual reference
    └── plot_46_FIC_103A.png
```

---

## Usage

### 1. Interpolation (`interpolation/`)

```bash
# Query valve opening (%) -> Cv & flowrate
python interpolation/run_interpolation.py --opening 22.5
python interpolation/run_interpolation.py --method spline --opening 22.5

# Run linear interpolation pipeline (writes to InfluxDB: predicted_fuel_gas_103a)
python interpolation/run_interpolation.py --run-pipeline
```

### 2. Machine Learning (`machine_learning/`)

```bash
# Run end-to-end training, inference, & evaluation
python machine_learning/run_pipeline.py

# Or run single latest point inference (for 5-min cron, EWMA disabled)
python machine_learning/src/inference.py --single-point
```

- **7 Raw Features (`SELECTED_FEATURE_TAGS`)**: `46-TIC-101.A`, `46-TI-102.A`, `46-PIC-109.A`, `46-TIC-106`, `46-FIC-101_PV`, `46-AI-101.A`, `46-FIC-103A1.Q`.
- **InfluxDB Outputs (`whb_medco`)**: `softsensor_fuelgas_output` (5-minute) and `softsensor_fuelgas_daily` (daily aggregation).
