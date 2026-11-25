"""
Configuration file for Fuel Gas Soft Sensor (46-FIC-103A1.A).
Self-contained within fuel_gas/machine_learning.
"""

import os

# Base Directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(ML_DIR)
VSENSOR_ROOT = os.path.dirname(PROJECT_ROOT)

# Data, Model & Report Directories
DATA_DIR = os.path.join(VSENSOR_ROOT, "data")
LOCAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(ML_DIR, "models")
REPORTS_DIR = os.path.join(ML_DIR, "reports")
FIG_DIR = os.path.join(REPORTS_DIR, "figures")

# InfluxDB Configuration
INFLUX_DATABASE = os.getenv("INFLUX_DATABASE", "whb_medco")
HISTORIAN_TABLE = os.getenv("INFLUX_MEASUREMENT", "historian_clean")
OUTPUT_MEASUREMENT = "softsensor_fuelgas_output"
OUTPUT_DAILY_MEASUREMENT = "softsensor_fuelgas_daily"

# DuckDB Legacy Reference Paths (Retained for offline fallback if needed)
DB_HISTORIAN_PATH = os.path.join(DATA_DIR, "whb1_data2023-2026.duckdb")
TRAIN_DATA_CSV = os.path.join(LOCAL_DATA_DIR, "fuel_valid_train_data.csv")
TEST_DATA_CSV = os.path.join(LOCAL_DATA_DIR, "fuel_valid_test_data.csv")
VALID_DATA_CSV = TRAIN_DATA_CSV

DB_OUTPUT_PATH = os.path.join(DATA_DIR, "whb1_fuelgas_softsensor_output.duckdb")
MODEL_PATH = os.path.join(MODELS_DIR, "softsensor_103a1.joblib")
SUMMARY_JSON_PATH = os.path.join(MODELS_DIR, "training_evaluation_summary.json")

for _d in (MODELS_DIR, DATA_DIR, REPORTS_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)

# Target & Primary Control Valve Tags
TARGET_TAG = "46-FIC-103A1.A"       # Fuel Gas Flow 1st Stage PV [LBS/HR]
VALVE_OPENING_TAG = "46-FIC-103A1.Q" # Fuel Gas Control Valve 46-FV-103A Opening [%]

# Verified 7 raw feature tags selected for training & inference.
SELECTED_FEATURE_TAGS = [
    "46-TIC-101.A",    # WHB #1 Flue Gas / Incinerator Temperature [deg F]
    "46-TI-102.A",     # Feedwater Temperature [deg F]
    "46-PIC-109.A",    # Steam Drum Pressure PV [PSIG]
    "46-TIC-106",      # Process Temperature [deg F]
    "46-FIC-101_PV",   # Incinerator Feed Flow PV [MMSCFD]
    "46-AI-101.A",     # Excess O2 Analyzer [%]
    "46-FIC-103A1.Q",  # Fuel Gas Control Valve 46-FV-103A Opening [%]
]

REQUIRED_RAW_TAGS = list(SELECTED_FEATURE_TAGS)
RANDOM_SEED = 42

# Operational Gating & Sensor Drift Alarm Parameters
MIN_VALVE_OPENING = 0.5   # Minimum valve opening (%) considered active
MIN_FLUE_TEMP = 300.0     # Minimum WHB #1 exit temp (deg F) for active flame
MAX_STACK_O2 = 17.5       # Maximum stack O2 (%) for active combustion
MIN_ACTIVE_FLOW = 500.0   # Minimum flow (LBS/HR) for active MAPE & relative drift
DRIFT_REL_THRESHOLD = 0.15
EWMA_ALPHA = 0.05

