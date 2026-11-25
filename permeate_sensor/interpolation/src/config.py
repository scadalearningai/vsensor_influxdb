"""
Configuration and centralized paths for the Interpolation module.
"""

import os

# Base paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.dirname(SRC_DIR)
PROJECT_ROOT = os.path.dirname(MODULE_ROOT)
VSENSOR_ROOT = os.path.dirname(PROJECT_ROOT)

# Data paths (pointing to root vsensor/data)
DATA_DIR = os.path.join(VSENSOR_ROOT, 'data')
LOCAL_DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
DEFAULT_PERMEATE_CSV = os.path.join(DATA_DIR, 'cv_permeate.csv') if os.path.exists(os.path.join(DATA_DIR, 'cv_permeate.csv')) else os.path.join(LOCAL_DATA_DIR, 'cv_permeate.csv')
DEFAULT_FUELGAS_CSV = os.path.join(DATA_DIR, 'cv_fuelgas.csv') if os.path.exists(os.path.join(DATA_DIR, 'cv_fuelgas.csv')) else os.path.join(LOCAL_DATA_DIR, 'cv_fuelgas.csv')
# InfluxDB Configuration
INFLUX_DATABASE = os.getenv("INFLUX_DATABASE", "whb_medco")
HISTORIAN_TABLE = os.getenv("INFLUX_MEASUREMENT", "historian_clean")

# Legacy DuckDB paths
DB_PERMEATE_PATH = os.path.join(DATA_DIR, 'whb1_data_28102026.duckdb') if os.path.exists(os.path.join(DATA_DIR, 'whb1_data_28102026.duckdb')) else os.path.join(LOCAL_DATA_DIR, 'whb1_permeate.duckdb')
DB_SOFTSENSOR_PATH = os.path.join(DATA_DIR, 'whb1_softsensor_output.duckdb')

# Reports and Figures
REPORTS_DIR = os.path.join(MODULE_ROOT, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
