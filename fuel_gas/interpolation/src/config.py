"""
Configuration and centralized paths for the Fuel Gas Interpolation module.
Supports both Piecewise Linear (DCS/SCADA standard) and Natural Cubic Spline models.
"""

import os

# Base Directories
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.dirname(SRC_DIR)
PROJECT_ROOT = os.path.dirname(MODULE_ROOT)
VSENSOR_ROOT = os.path.dirname(PROJECT_ROOT)

# Data Paths
DATA_DIR = os.path.join(VSENSOR_ROOT, "data")
LOCAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data")

_cv_local = os.path.join(LOCAL_DATA_DIR, "cv_fuelgas.csv")
_cv_root = os.path.join(DATA_DIR, "cv_fuelgas.csv")
DEFAULT_CV_PATH = _cv_local if os.path.exists(_cv_local) else _cv_root

# InfluxDB Configuration (Historian Input)
INFLUX_DATABASE = os.getenv("INFLUX_DATABASE", "whb_medco")
HISTORIAN_TABLE = os.getenv("INFLUX_MEASUREMENT", "historian_clean")

# TimescaleDB Configuration (Output)
TIMESCALE_DATABASE = os.getenv("TIMESCALE_DATABASE", "whb_medco")
TIMESCALE_SCHEMA = os.getenv("TIMESCALE_SCHEMA", "public")
OUTPUT_TABLE_LINEAR = "predicted_fuel_gas_103a"

# Legacy DuckDB Reference Paths
LOCAL_TAG_DB_PATH = os.path.join(LOCAL_DATA_DIR, "tag_identification_data.duckdb")
_db_master = os.path.join(DATA_DIR, "whb1_data_28102026.duckdb")
DEFAULT_DB_PATH = _db_master if os.path.exists(_db_master) else LOCAL_TAG_DB_PATH

# Reports, Figures & Slides Paths
REPORTS_DIR = os.path.join(MODULE_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
SLIDES_DIR = os.path.join(MODULE_ROOT, "slides")

for _d in (REPORTS_DIR, FIGURES_DIR, SLIDES_DIR):
    os.makedirs(_d, exist_ok=True)

# Primary Tags & Constants
TARGET_TAG = "46-FIC-103A1.A"
VALVE_OPENING_TAG = "46-FIC-103A1.Q"
CV_TO_FLOW_FACTOR = 48.4467
MIN_ACTIVE_OPENING = 0.5

