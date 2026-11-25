# Decoupled Physical Soft Sensor (`46-FQI-106A`)

Physics-based soft sensor module for estimating permeate gas fuel flow (**`46-FQI-106A`**) on Waste Heat Boiler #1 (WHB #1), Medco E&P Indonesia, connected to **InfluxDB v1.x**.

---

## Directory Structure

1. **[`interpolation/`](interpolation/README.md)** — **Valve Characteristics Interpolation** based on `data/cv_permeate.csv` (*Two-Stage Linear Estimator* and *Natural Cubic Spline*), plus hourly evaluation directly from InfluxDB.
2. **[`machine_learning/`](machine_learning/)** — Production **LightGBM Decoupled Soft Sensor (9 Stateless Features)** pipeline.

```text
permeate_sensor/
├── README.md
├── data/                                    # Valve curve tables & valid training intervals
│   ├── cv_fuelgas.csv
│   ├── cv_permeate.csv
│   └── permeate_valid_data.csv
├── interpolation/                           # [METHOD 1] Valve Characteristics Interpolation
│   ├── README.md
│   ├── __init__.py
│   ├── run_spline.py                        # Cubic Spline CLI (query, export, plot)
│   └── src/
│       ├── __init__.py
│       ├── config.py                        # InfluxDB config, paths, & constants
│       ├── estimator.py                     # PermeateFlowEstimator (Two-Stage)
│       ├── evaluate_hourly.py               # 24-hour evaluation from InfluxDB
│       ├── test_estimator.py                # Unit tests for PermeateFlowEstimator
│       └── valve_spline.py                  # ValveSplineInterpolator (Cubic Spline)
├── machine_learning/                        # [METHOD 2] LightGBM Soft Sensor
│   ├── run_pipeline.py                      # End-to-end runner (train -> inference -> evaluate)
│   ├── src/
│   │   ├── __init__.py
│   │   ├── config.py                        # InfluxDB config, sensor tags, & drift thresholds
│   │   ├── features.py                      # 9 stateless features & whb1_on gating
│   │   ├── train.py                         # LightGBM training (TimeSeriesSplit + Out-of-Time)
│   │   ├── inference.py                     # SoftSensorEngine & CLI (--single-point, --latest, etc.)
│   │   └── evaluate.py                      # Metrics evaluation & Matplotlib figures from InfluxDB
│   └── models/
│       ├── softsensor_106a.joblib           # Trained LightGBM model artifact
│       └── retrain_evaluation_summary.json
└── references/                              # Reference diagrams & daily report
```

---

## Running the Pipelines

### 1. Machine Learning (`machine_learning/`)

```bash
cd machine_learning

# Run full pipeline (train -> inference -> evaluate)
python run_pipeline.py

# Or run individual steps:
python src/train.py
python src/inference.py                  # Full historian -> InfluxDB
python src/inference.py --single-point   # 1 latest point for 5-min cron (EWMA disabled)
python src/evaluate.py
```

- **Architecture**: **9 Stateless Physical Features** (`46-FIC-106A.Q`, `54-FI-1002`, `20-PIC-106.A`, `46-FQI-206A.A`, `24-FQI-0124`, `24-FQI-0123`, `15-FQI-002.A`, `15-FQI-402.A`, `46-TIC-101.A`) using `LGBMRegressor`.
- **InfluxDB Outputs (`whb_medco`)**: `softsensor_permeate_output` (5-minute) and `softsensor_permeate_daily` (daily aggregation).

### 2. Interpolation (`interpolation/`)

```bash
cd interpolation
python run_spline.py --valve permeate --opening 45.5
python src/evaluate_hourly.py --date 2026-09-09 --mode mean
```
