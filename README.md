# WHB #1 Virtual Sensor System (InfluxDB Edition)

A *Virtual Sensor / Soft Sensor* system for **Waste Heat Boiler #1 (WHB #1)**, Medco E&P Indonesia, powered by **InfluxDB v1.x (Time-Series Database)**. This project provides real-time and batch flowrate estimation as well as *sensor drift* detection for two critical sensors: **Fuel Gas (`fuel_gas/`)** and **Permeate Gas (`permeate_sensor/`)**.

---

## 1. Module Purpose & Objectives

### A. Fuel Gas Soft Sensor (`fuel_gas/`, Target: `46-FIC-103A1.A`)
- **Purpose:** Independently estimates the Stage-1 *fuel gas* flowrate to the WHB #1 burner without relying on the physical flow sensor `46-FIC-103A1.A`.
- **Objectives:** Provides a virtual backup reading when the physical sensor experiences drift or calibration issues, enforces *physics gating* (`0.0` when the burner is off or the valve is closed), and triggers `drift_alarm` when physical sensor deviation exceeds the threshold.

### B. Permeate Gas Soft Sensor (`permeate_sensor/`, Target: `46-FQI-106A.A`)
- **Purpose:** Estimates the *permeate* gas flowrate from the membrane units to WHB #1 using upstream (feed & membrane) and downstream (WHB & blowdown) process signals decoupled from the target sensor `46-FQI-106A.A`.
- **Objectives:** Detects permeate flowmeter drift early, maintains daily mass-balance consistency (`softsensor_permeate_daily`), and provides *stateless* predictions ready for 5-minute SCADA/cron execution.

---

## 2. Models, Features, & Interpolation Architecture

### A. Fuel Gas (`fuel_gas/`)
1. **Method 1 — Valve Characteristics Interpolation (`fuel_gas/interpolation/`)**
   - *First-principles* estimation based on control valve opening (`46-FIC-103A1.Q`) and the valve characteristic curve ([`cv_fuelgas.csv`](fuel_gas/data/cv_fuelgas.csv)):
     - **Piecewise Linear (`ValveFlowEstimator`):** Maps valve opening (%) → capacity coefficient $C_v$ via linear interpolation (`numpy.interp`), then computes $\text{Flow} = 48.4467 \times C_v$. Compatible with PLC/DCS/SCADA function blocks and writes batch results to the `predicted_fuel_gas_103a` measurement in InfluxDB.
     - **Natural Cubic Spline (`ValveSplineInterpolator`):** Smooth spline curve supporting both *forward* queries (opening → flow & $C_v$) and *inverse* queries (target flow → required opening).
2. **Method 2 — Machine Learning (`fuel_gas/machine_learning/`)**
   - **7 Raw Features (Stateless, zero target/setpoint leakage):**
     - `46-TIC-101.A` (Incinerator Flue Gas Temperature)
     - `46-TI-102.A` (Feedwater Temperature)
     - `46-PIC-109.A` (Steam Drum Pressure PV)
     - `46-TIC-106` (Process Temperature)
     - `46-FIC-101_PV` (Incinerator Feed Flow PV)
     - `46-AI-101.A` (Excess O2 Analyzer)
     - `46-FIC-103A1.Q` (Fuel Gas Control Valve Opening)
   - **Physics Gating:** `whb1_on`, `valve_open`, and `fuel_gas_active`.
   - **Models:** Evaluates 4 candidate architectures using *GroupKFold* across operational intervals (`StackedEnsemble` [out-of-fold stacking of `RandomForest`, `XGBoost`, `MLP` + `RidgeCV` meta-learner], `RandomForest`, `XGBoost`, `MLP`). The selected model (`StackedEnsemble`) is saved to `fuel_gas/machine_learning/models/softsensor_103a1.joblib`.

### B. Permeate Sensor (`permeate_sensor/`)
1. **Method 1 — Valve Characteristics Interpolation (`permeate_sensor/interpolation/`)**
   - Uses the valve characteristic table [`cv_permeate.csv`](permeate_sensor/data/cv_permeate.csv):
     - **Two-Stage Estimator (`PermeateFlowEstimator`):** Stage 1 maps valve opening (%) → $C_v$ via linear interpolation; Stage 2 computes $Q = m \cdot C_v + c$.
     - **Cubic Spline (`ValveSplineInterpolator`, `run_spline.py`):** Smooth spline interpolation for *forward*/*inverse* queries, CSV grid export, and curve plotting.
     - **Hourly Evaluation (`src/evaluate_hourly.py`):** Queries daily data (`46-FIC-106A.Q` / `46-FV-106A` and `46-FIC-106A.A`) directly from InfluxDB for `mean` or `snapshot` evaluation.
2. **Method 2 — Machine Learning (`permeate_sensor/machine_learning/`)**
   - **9 Stateless Features (Units 15, 20, 24, 46, 54):**
     - `46-FIC-106A.Q` (Permeate Control Valve Opening [%])
     - `54-FI-1002` (Boiler Steam Blowdown Flow)
     - `20-PIC-106.A` (Permeate Header Pressure [PSIG])
     - `46-FQI-206A.A` (Permeate Flow to WHB #2 [LBS/HR])
     - `24-FQI-0124` (Membrane Train E/F Inlet Flow)
     - `24-FQI-0123` (Membrane Train E/F Residue Flow)
     - `15-FQI-002.A` (Plant Feed Gas Line 1)
     - `15-FQI-402.A` (Plant Feed Gas Line 2)
     - `46-TIC-101.A` (WHB #1 Flue Gas Temp [deg F])
   - **Physics Gating:** `whb1_on` (active when `46-TIC-101.A > 300.0` and `46-AI-101.A < 17.5` or NaN).
   - **Model:** `LGBMRegressor` (LightGBM) validated chronologically using *5-Fold Expanding Window TimeSeriesSplit* and *Out-of-Time Holdout Split*, saved to `permeate_sensor/machine_learning/models/softsensor_106a.joblib`.

---

## 3. Directory Structure

```text
vsensor_influxdb/
├── common/
│   └── influx_client.py               # InfluxDB v1.x client & helpers (query & line protocol write)
├── docker/
│   ├── crontab                        # 5-minute cron schedule (:30s) for permeate & fuel_gas
│   └── entrypoint.sh                  # Exports env vars to /etc/environment & starts cron
├── fuel_gas/
│   ├── data/                          # cv_fuelgas.csv, fuel_valid_train_data.csv, fuel_valid_test_data.csv
│   ├── interpolation/                 # Linear & Cubic Spline interpolation (run_interpolation.py)
│   └── machine_learning/              # src/{config,features,models,train,inference,evaluate}.py & models/
├── permeate_sensor/
│   ├── data/                          # cv_permeate.csv, permeate_valid_data.csv
│   ├── interpolation/                 # Two-Stage & Cubic Spline (run_spline.py, src/evaluate_hourly.py)
│   └── machine_learning/              # src/{config,features,train,inference,evaluate}.py & models/
├── .dockerignore
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── README.md
└── requirements.txt
```

---

## 4. Database Configuration & Deployment Overrides

### A. Changing Host, Port, User, Password, & Database (`.env` / Environment Variables)
When deploying to different environments (staging/production), update the configuration without modifying Python code using any of the following methods:

1. **Via `.env` file in the project root** (copy from [`.env.example`](.env.example)):
   ```env
   INFLUX_HOST=192.168.5.216
   INFLUX_PORT=8086
   INFLUX_USER=spcuser
   INFLUX_PASSWORD=Kembar12345
   INFLUX_DATABASE=whb_medco
   INFLUX_MEASUREMENT=historian_clean
   INFLUX_CHRONOGRAF_PORT=8888
   ```
   - [`common/influx_client.py`](common/influx_client.py) loads `.env` automatically at startup via `os.environ.setdefault(...)`.
   - System/OS environment variables take **precedence** over `.env`.

2. **Override via Environment Variables (Docker / Shell):**
   ```bash
   # Linux / macOS
   export INFLUX_HOST=10.10.20.5
   export INFLUX_PORT=8086
   export INFLUX_DATABASE=whb_prod

   # Or for ad-hoc Docker runs
   docker run --rm --env-file .env \
     -e INFLUX_HOST=10.10.20.5 \
     -e INFLUX_PORT=8086 \
     -e INFLUX_DATABASE=whb_prod \
     vsensor-influxdb:latest python permeate_sensor/machine_learning/src/inference.py --single-point
   ```
   In `docker-compose.yml`, you can also add an `environment:` block under `vsensor-cron` to override `.env` values per deployment environment.

3. **Changing Output Measurement Names & Drift Parameters (`src/config.py`):**
   - [`fuel_gas/machine_learning/src/config.py`](fuel_gas/machine_learning/src/config.py): `OUTPUT_MEASUREMENT`, `OUTPUT_DAILY_MEASUREMENT`, `DRIFT_REL_THRESHOLD`, `EWMA_ALPHA`, `MIN_ACTIVE_FLOW`.
   - [`permeate_sensor/machine_learning/src/config.py`](permeate_sensor/machine_learning/src/config.py): `OUTPUT_MEASUREMENT`, `OUTPUT_DAILY_MEASUREMENT`, `DRIFT_REL_THRESHOLD`, `EWMA_ALPHA`, `MIN_ACTIVE_FLOW`.
   - [`fuel_gas/interpolation/src/config.py`](fuel_gas/interpolation/src/config.py): `OUTPUT_TABLE_LINEAR`.

4. **InfluxDB User Permissions:**
   - **Training & Evaluation:** Requires **READ** access to `historian_clean` (and output measurements for `fuel_gas/machine_learning/src/evaluate.py`).
   - **Inference & Cron:** Requires **READ** (`historian_clean`) and **WRITE** access on `INFLUX_DATABASE`. `write_dataframe()` calls `ensure_database_exists()` to create the database automatically if it does not exist.

### B. Data Access Layer (`common/influx_client.py`)
- `get_client()`: Initializes an authenticated `InfluxDBClient`.
- `query_dataframe(query)`: Executes InfluxQL and returns a pandas `DataFrame` with a normalized `Timestamp` column.
- `query_interval_tags(tags, start, end)`: Queries process tags over a specific time range.
- `query_multiple_intervals(tags, intervals_df)`: Queries and concatenates data across multiple training/test windows.
- `write_dataframe(df, measurement)`: Writes a `DataFrame` in batches using InfluxDB Line Protocol.

### C. InfluxDB Measurements (`whb_medco`)
1. **`historian_clean`** *(5-min SCADA Input)*: 24+ primary process tags.
2. **`predicted_fuel_gas_103a`** *(Fuel Gas Interpolation Output)*: `46-FIC-103A.Q_opening`, `estimated_cv_linear`, `predicted_flow_linear`, `sensor_actual_flow_uncalibrated`, `sensor_drift_deviation`.
3. **`softsensor_fuelgas_output`** *(Fuel Gas ML Output)*: `measured_103A1`, `valve_opening_pct`, `whb1_on`, `valve_open`, `fuel_gas_active`, `y_hat`, `F_hat`, `deviation`, `rel_deviation`, `rel_dev_ewma`, `drift_alarm`, `trusted`, `suspect`.
4. **`softsensor_fuelgas_daily`** *(Fuel Gas Daily Aggregation)*: `measured_total_lbs`, `softsensor_total_lbs`, `diff_pct`, `drift_alarm_frac`, `active_minutes`.
5. **`softsensor_permeate_output`** *(Permeate Gas ML Output)*: `measured_106A`, `whb1_on`, `feed_total`, `P_total`, `y_hat`, `F_hat`, `deviation`, `rel_deviation`, `rel_dev_ewma`, `drift_alarm`, `trusted`, `suspect`.
6. **`softsensor_permeate_daily`** *(Permeate Gas Daily Aggregation)*: `measured_total_lbs`, `softsensor_total_lbs`, `diff_%`, `drift_alarm_frac`.

---

## 5. Running Pipelines, Inference, & Cron

### A. Fuel Gas (`fuel_gas/`)
```bash
# 1. Valve characteristics interpolation
cd fuel_gas/interpolation
python run_interpolation.py --opening 22.5
python run_interpolation.py --method spline --flowrate 440
python run_interpolation.py --run-pipeline          # Batch to InfluxDB (predicted_fuel_gas_103a)

# 2. Machine Learning (Training, Inference, Evaluation)
cd ../machine_learning
python src/train.py                                 # Retrain model from fuel_valid_train_data.csv
python src/inference.py                             # Full historian batch inference to InfluxDB
python src/inference.py --single-point              # Single latest point inference (EWMA disabled)
python src/evaluate.py                              # Evaluation & figures to reports/
python run_pipeline.py                              # Run train -> inference -> evaluate
```

### B. Permeate Sensor (`permeate_sensor/`)
```bash
# 1. Interpolation & hourly evaluation
cd permeate_sensor/interpolation
python run_spline.py --valve permeate --opening 45.5
python src/evaluate_hourly.py --date 2026-09-09 --mode mean

# 2. Machine Learning (Training, Inference, Evaluation)
cd ../machine_learning
python src/train.py                                 # Retrain model from permeate_valid_data.csv
python src/inference.py                             # Full historian batch inference to InfluxDB
python src/inference.py --single-point              # Single latest point inference (EWMA disabled)
python src/evaluate.py                              # Evaluation & figures to reports/
python run_pipeline.py                              # Run train -> inference -> evaluate
```

### C. `src/inference.py` CLI Options (Both `fuel_gas` & `permeate_sensor`)
| Flag | Description |
|---|---|
| *(no flags)* | Full-historian batch inference on `historian_clean` + daily aggregation |
| `--single-point` | Query **1 latest point** (`ORDER BY time DESC LIMIT 1`); implies `--disable-ewma` & `--skip-daily` |
| `--latest` | Query last 180 minutes (`now() - 180m`), write last 5 minutes, skip daily |
| `--disable-ewma` | Disable *EWMA smoothing*; `drift_alarm` uses instantaneous relative deviation |
| `--lookback-minutes N` | Query `N` minutes back from InfluxDB |
| `--write-minutes N` | Write only points within the last `N` minutes of the latest timestamp (`0` = latest point only) |
| `--delay-minutes N` | Shift target end time back by `N` minutes from `now()` (e.g., `30` when cron at `05:30` targets `05:00`) |
| `--end-time "YYYY-MM-DD HH:MM:SS"` | Explicit target end timestamp |
| `--skip-daily` | Skip writing the daily aggregation measurement |

### D. Host Cron Examples (Every 5 Minutes at `:30` Seconds)
**Linux (`crontab -e`):**
```cron
*/5 * * * * sleep 30 && cd /path/to/vsensor_influxdb/permeate_sensor/machine_learning && /path/to/venv/bin/python src/inference.py --single-point >> /var/log/permeate_inference.log 2>&1
*/5 * * * * sleep 30 && cd /path/to/vsensor_influxdb/fuel_gas/machine_learning && /path/to/venv/bin/python src/inference.py --single-point >> /var/log/fuelgas_inference.log 2>&1
```
**Windows (Task Scheduler):**
- Set *Trigger* to start at `00:00:30` and *Repeat task every: 5 minutes*.
- Set *Action* to run `python.exe src/inference.py --single-point` with *Start in* set to `permeate_sensor\machine_learning` (and a second task for `fuel_gas\machine_learning`).

### E. Training on Specific Historical Date Ranges
- **Permeate Sensor:** Edit `Mulai` & `Selesai` intervals (`%d-%b-%y %H:%M`) in [`permeate_sensor/data/permeate_valid_data.csv`](permeate_sensor/data/permeate_valid_data.csv), and update `train_mask` / `val_mask` / `test_mask` in `permeate_sensor/machine_learning/src/train.py` if the year range changes.
- **Fuel Gas:** Edit date intervals in [`fuel_gas/data/fuel_valid_train_data.csv`](fuel_gas/data/fuel_valid_train_data.csv) and [`fuel_gas/data/fuel_valid_test_data.csv`](fuel_gas/data/fuel_valid_test_data.csv).

---

## 6. Running with Docker

The Docker image is configured to run 5-minute cron inference (`--single-point` at `:30s`) for both **`permeate_sensor`** and **`fuel_gas`**, and also supports ad-hoc commands:

```bash
# 1. Build image
docker build -t vsensor-influxdb:latest .

# 2. Start 5-minute cron daemon in the background (uses root .env)
docker compose up -d
docker logs -f vsensor-cron

# 3. Stop or rebuild cron service
docker compose down
docker compose up -d --build   # Rebuild after modifying docker/crontab or source code

# 4. Run ad-hoc commands (training / evaluation / interpolation)
docker run --rm --env-file .env vsensor-influxdb:latest python permeate_sensor/machine_learning/src/train.py
docker run --rm --env-file .env vsensor-influxdb:latest python fuel_gas/machine_learning/src/train.py
docker run --rm --env-file .env vsensor-influxdb:latest python fuel_gas/interpolation/run_interpolation.py --opening 22.5
```

- **Changing Docker cron schedule:** Edit [`docker/crontab`](docker/crontab) (e.g., switch `--single-point` to `--latest`), then run `docker compose up -d --build`.
- **Changing container timezone:** Update `TZ=Asia/Jakarta` in [`Dockerfile`](Dockerfile) or override via `-e TZ=UTC` / `environment:` in `docker-compose.yml`.

---

## 7. Dependencies

Dependencies are listed in [`requirements.txt`](requirements.txt):
```bash
pip install -r requirements.txt
```
Includes:
- **InfluxDB & Data Science:** `influxdb>=5.3.2`, `pandas>=2.0.0`, `numpy>=1.24.0`, `scipy>=1.11.0`, `scikit-learn>=1.3.0`, `lightgbm>=4.0.0`, `xgboost>=2.0.0`, `joblib>=1.3.0`, `matplotlib>=3.7.0`, `seaborn>=0.13.0`, `python-dotenv>=1.0.0`
- **Additional SQL & Time-Series Connectors:** `duckdb>=0.9.0`, `psycopg2-binary>=2.9.9` (PostgreSQL / TimescaleDB), `pymssql>=2.2.11` & `pyodbc>=5.1.0` (SQL Server), `sqlalchemy>=2.0.0`
