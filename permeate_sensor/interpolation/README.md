# Control Valve Interpolation Module (`permeate_sensor/interpolation`)

This module provides flowrate estimation based on control valve characteristics from `../data/cv_permeate.csv` and `../data/cv_fuelgas.csv`, supporting:
1. **Natural Cubic Spline (`ValveSplineInterpolator`)** — uses `scipy.interpolate.CubicSpline` to model the non-linear relationship between valve opening (%), flow coefficient ($C_v$), and mass flowrate.
2. **Two-Stage Estimator (`PermeateFlowEstimator`)** — maps `opening (%) -> Cv` via linear interpolation and `Cv -> flowrate` via $Q = m \cdot C_v + c$, plus 24-hour evaluation directly from **InfluxDB** (`src/evaluate_hourly.py`).

---

## Directory Structure

```text
interpolation/
├── README.md
├── __init__.py
├── run_spline.py                        # Unified CLI runner (query, export, plot)
└── src/
    ├── __init__.py
    ├── config.py                        # InfluxDB config, CSV paths, & output directories
    ├── estimator.py                     # PermeateFlowEstimator (Two-Stage)
    ├── evaluate_hourly.py               # 24-hour evaluation (mean / snapshot) from InfluxDB
    ├── test_estimator.py                # Unit tests for PermeateFlowEstimator
    └── valve_spline.py                  # ValveSplineInterpolator & factory functions
```

---

## Usage Guide

### 1. Cubic Spline CLI (`run_spline.py`)

```bash
# Forward query: estimate flowrate & Cv from valve opening (%)
python interpolation/run_spline.py --valve permeate --opening 45.5

# Inverse query: required valve opening (%) for a target flowrate
python interpolation/run_spline.py --valve permeate --flowrate 250000

# Query for fuel gas valve
python interpolation/run_spline.py --valve fuelgas --opening 50.0

# Export fine-grid CSV & plot curves to reports/figures/
python interpolation/run_spline.py --all
```

### 2. Hourly Evaluation from InfluxDB (`src/evaluate_hourly.py`)

```bash
# Hourly mean evaluation for a specific date from historian_clean
python interpolation/src/evaluate_hourly.py --date 2026-09-09 --mode mean

# Snapshot evaluation at :00 minutes
python interpolation/src/evaluate_hourly.py --date 2026-09-09 --mode snapshot
```

### 3. Python API

```python
from interpolation.src import get_permeate_spline, get_fuelgas_spline

spline = get_permeate_spline(bc_type='natural')
flow = spline.predict_flowrate(50.0)
cv = spline.predict_cv(50.0)
gain = spline.derivative_flowrate(50.0)
opening = spline.predict_opening(250000)
```
