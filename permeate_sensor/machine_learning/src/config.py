"""
Configuration file for Decoupled Physical Soft Sensor (46-FQI-106A).
Self-contained within the machine_learning module.
"""

import os

# Base Directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(ML_DIR)
VSENSOR_ROOT = os.path.dirname(PROJECT_ROOT)

# Data & Model Directories (pointing to root vsensor/data)
DATA_DIR = os.path.join(VSENSOR_ROOT, 'data')
LOCAL_DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(ML_DIR, 'models')
REPORTS_DIR = os.path.join(ML_DIR, 'reports')
FIG_DIR = os.path.join(REPORTS_DIR, 'figures')
PLOTLY_DIR = os.path.join(REPORTS_DIR, 'plotly')

# InfluxDB Configuration (Historian Input)
INFLUX_DATABASE = os.getenv("INFLUX_DATABASE", "whb_medco")
HISTORIAN_TABLE = os.getenv("INFLUX_MEASUREMENT", "historian_clean")

# TimescaleDB Configuration (Soft Sensor Output)
TIMESCALE_DATABASE = os.getenv("TIMESCALE_DATABASE", "whb_medco")
TIMESCALE_SCHEMA = os.getenv("TIMESCALE_SCHEMA", "public")
OUTPUT_MEASUREMENT = "softsensor_permeate_output"
OUTPUT_DAILY_MEASUREMENT = "softsensor_permeate_daily"

# DuckDB Legacy Reference Paths (Retained for offline fallback if needed)
DB_HISTORIAN_2023_2026 = os.path.join(DATA_DIR, 'whb1_data2023-2026.duckdb')
DB_HISTORIAN_28102026 = os.path.join(DATA_DIR, 'whb1_data_28102026.duckdb')
DB_HISTORIAN_PATH = DB_HISTORIAN_2023_2026 if os.path.exists(DB_HISTORIAN_2023_2026) else DB_HISTORIAN_28102026
VALID_DATA_CSV = os.path.join(DATA_DIR, 'permeate_valid_data.csv') if os.path.exists(os.path.join(DATA_DIR, 'permeate_valid_data.csv')) else os.path.join(LOCAL_DATA_DIR, 'permeate_valid_data.csv')

DB_INPUT_PATH = DB_HISTORIAN_PATH if os.path.exists(DB_HISTORIAN_PATH) else os.path.join(DATA_DIR, 'whb1_permeate.duckdb')
DB_OUTPUT_PATH = os.path.join(DATA_DIR, 'whb1_softsensor_output.duckdb')
MODEL_PATH = os.path.join(MODELS_DIR, 'softsensor_106a.joblib')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PLOTLY_DIR, exist_ok=True)

# Target Tag
TARGET_TAG = '46-FQI-106A.A'

# Decoupled Physical Features (12 Features from Membrane & WHB subsystems)
MEMBRANE_FEED_TAGS = [
    '15-FQI-002.A', '15-FQI-402.A',  # Raw feed
    '20-FQI-101.A', '20-FQI-102.A',  # Train A/B
    '20-FQI-201.A', '20-FQI-202.A',  # Train C/D
    '24-FQI-0124', '24-FQI-0123',    # Train E/F
    '24-FQI-0224', '24-FQI-0223'     # Train G/H
]
WHB_TAGS = [
    '46-FQI-206A.A',  # Permeate flow to WHB #2
    '46-TIC-101.A',   # WHB #1 Flue Gas Temp
    '46-AI-101.A'     # WHB #1 Flue Gas O2
]
VALVE_PRESSURE_TAGS = [
    '46-FIC-106A.Q',  # Permeate Control Valve Opening [%]
    '20-PIC-106.A'    # Permeate Header Pressure [PSIG]
]

ALL_RAW_TAGS = MEMBRANE_FEED_TAGS + WHB_TAGS + VALVE_PRESSURE_TAGS + [TARGET_TAG]

# Sensor Drift Alarm Parameters
DRIFT_REL_THRESHOLD = 0.15      # 15% threshold for persistent relative deviation
EWMA_ALPHA = 0.05               # Smoothing factor for EWMA (~30-60 min response window)
MIN_ACTIVE_FLOW = 500.0         # Minimum flow (LBS/HR) to trigger active relative drift evaluation
