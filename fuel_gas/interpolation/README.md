# Fuel Gas Control Valve Interpolation Module (`46-FV-103A`)

This module provides fuel gas flowrate prediction based on control valve design characteristics from `../data/cv_fuelgas.csv`, supporting:
1. **Piecewise Linear Interpolation (`ValveFlowEstimator`)** — PLC/DCS/SCADA standard (`Flow = 48.4467 * Cv`), plus drift analysis against the physical sensor `46-FIC-103A1.A` reading from and writing to **InfluxDB**.
2. **Natural Cubic Spline (`ValveSplineInterpolator`)** — smooth $C^2$ curve supporting *forward* (`opening -> flowrate, Cv, gain`) and *inverse* (`flowrate -> opening`) queries.

---

## Directory Structure

```text
interpolation/
├── README.md
├── __init__.py
├── run_interpolation.py                 # Unified CLI runner (linear & cubic spline)
├── src/
│   ├── __init__.py
│   ├── config.py                        # InfluxDB config, paths, & centralized constants
│   ├── estimator.py                     # ValveFlowEstimator & run_prediction_pipeline (InfluxDB)
│   └── valve_spline.py                  # ValveSplineInterpolator & get_fuelgas_spline
└── scripts/
    └── plot_prediction_results.py       # Valve curve & drift plots from InfluxDB to reports/figures/
```

---

## Usage Guide

### 1. CLI Runner (`run_interpolation.py`)

```bash
# Full linear interpolation pipeline -> InfluxDB measurement predicted_fuel_gas_103a
python interpolation/run_interpolation.py --run-pipeline

# Dry-run without writing to InfluxDB + generate plots in reports/figures/
python interpolation/run_interpolation.py --no-save-db --plot

# Forward queries (Linear & Cubic Spline)
python interpolation/run_interpolation.py --opening 22.5
python interpolation/run_interpolation.py --method spline --opening 22.5

# Inverse query (Cubic Spline: target flowrate -> valve opening %)
python interpolation/run_interpolation.py --method spline --flowrate 440

# Plot Natural Cubic Spline curve to reports/figures/
python interpolation/run_interpolation.py --method spline --plot
```

### 2. Python API

```python
from interpolation import ValveFlowEstimator, get_fuelgas_spline

# Piecewise Linear (DCS standard)
est = ValveFlowEstimator()
print(est.predict_flow(22.5))  # -> ~438.44

# Natural Cubic Spline
spline = get_fuelgas_spline(bc_type="natural")
print(float(spline.predict_flowrate(22.5)))
print(float(spline.predict_opening(440.0)))
```
