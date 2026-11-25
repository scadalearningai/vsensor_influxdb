"""
First-Principles Fuel Gas Flow Estimator based on control valve Cv characteristics.
Uses Piecewise Linear interpolation (numpy.interp) compatible with PLC/DCS/SCADA function blocks.
"""

import os
import sys
import numpy as np
import pandas as pd

try:
    from .config import (
        CV_TO_FLOW_FACTOR,
        DEFAULT_CV_PATH,
        HISTORIAN_TABLE,
        INFLUX_DATABASE,
        MIN_ACTIVE_OPENING,
        OUTPUT_TABLE_LINEAR,
        PROJECT_ROOT,
        TARGET_TAG,
        VALVE_OPENING_TAG,
        VSENSOR_ROOT,
    )
except ImportError:
    from config import (
        CV_TO_FLOW_FACTOR,
        DEFAULT_CV_PATH,
        HISTORIAN_TABLE,
        INFLUX_DATABASE,
        MIN_ACTIVE_OPENING,
        OUTPUT_TABLE_LINEAR,
        PROJECT_ROOT,
        TARGET_TAG,
        VALVE_OPENING_TAG,
        VSENSOR_ROOT,
    )

if VSENSOR_ROOT not in sys.path:
    sys.path.insert(0, VSENSOR_ROOT)

from common.influx_client import query_dataframe
from common.timescale_client import write_dataframe as write_timescale_dataframe


class ValveFlowEstimator:
    """
    First-Principles Fuel Gas flowrate estimator based on control valve physical characteristics.
    Uses Piecewise Linear interpolation compatible with industrial SCADA/DCS standards.

    Assumptions:
    The physical flow sensor (46-FIC-103A1.A) is assumed to be inaccurate (experiencing sensor drift).
    Therefore, flowrate prediction is derived independently and objectively from:
    1. Valve opening percentage (46-FIC-103A1.Q)
    2. Valve capacity curve Cv (cv_fuelgas.csv) via point-to-point linear interpolation
    3. Capacity-to-flowrate relation: Flow = 48.4467 * Cv
    """

    def __init__(self, cv_csv_path: str = DEFAULT_CV_PATH):
        self.cv_csv_path = cv_csv_path
        self.cv_df = None
        self.openings = None
        self.cv_values = None
        self.cv_to_flow_factor = CV_TO_FLOW_FACTOR
        self._load_table()

    def _load_table(self) -> None:
        csv_file = self.cv_csv_path
        if not os.path.exists(csv_file):
            alt_file = os.path.join(PROJECT_ROOT, self.cv_csv_path)
            if os.path.exists(alt_file):
                csv_file = alt_file
            else:
                raise FileNotFoundError(f"Cv characteristic file not found at '{self.cv_csv_path}'")

        self.cv_df = pd.read_csv(csv_file).sort_values("opening").reset_index(drop=True)
        self.openings = self.cv_df["opening"].to_numpy(dtype=float)
        self.cv_values = self.cv_df["cv"].to_numpy(dtype=float)

        non_zero = self.cv_df[self.cv_df["cv"] > 0]
        if not non_zero.empty:
            self.cv_to_flow_factor = float((non_zero["flowrate"] / non_zero["cv"]).mean())

    def opening_to_cv(self, opening_pct):
        """Computes capacity coefficient Cv from valve opening percentage (0 - 100%)."""
        clamped = np.clip(np.asarray(opening_pct, dtype=float), 0.0, 100.0)
        cv = np.interp(clamped, self.openings, self.cv_values)
        return np.where(clamped <= 0.0, 0.0, cv)

    def predict_flow(self, opening_pct):
        """
        Stage 1: Valve Opening (%) -> Cv (via Linear Interpolation)
        Stage 2: Flowrate = 48.4467 * Cv
        """
        clamped = np.clip(np.asarray(opening_pct, dtype=float), 0.0, 100.0)
        cv = self.opening_to_cv(clamped)
        predicted_flow = self.cv_to_flow_factor * cv
        return np.where(clamped <= 0.0, 0.0, predicted_flow)

    def analyze_sensor_drift(self, opening_pct, sensor_flow) -> dict:
        """Analyzes actual physical sensor deviation against linear valve characteristic predictions."""
        opening_arr = np.asarray(opening_pct, dtype=float)
        sensor_arr = np.asarray(sensor_flow, dtype=float)
        pred_flow = self.predict_flow(opening_arr)
        mask = (opening_arr > MIN_ACTIVE_OPENING) & (~np.isnan(sensor_arr)) & (~np.isnan(pred_flow))

        sf = sensor_arr[mask]
        pf = pred_flow[mask]
        if len(sf) == 0:
            return {}

        deviation = sf - pf
        ratio = sf / np.maximum(pf, 1e-6)
        return {
            "Total Active Samples": int(len(sf)),
            "Mean Valve Predicted Flow (Linear)": round(float(np.mean(pf)), 2),
            "Mean Physical Sensor Reading": round(float(np.mean(sf)), 2),
            "Mean Deviation / Drift (Sensor - Predicted)": round(float(np.mean(deviation)), 2),
            "Median Deviation Ratio (Sensor / Predicted)": round(float(np.median(ratio)), 2),
            "Drift Standard Deviation": round(float(np.std(deviation)), 2),
        }


def run_prediction_pipeline(
    database: str = INFLUX_DATABASE,
    cv_csv_path: str = DEFAULT_CV_PATH,
    table_name: str = HISTORIAN_TABLE,
    save_to_db: bool = True,
):
    print("=" * 75)
    print("FUEL GAS FLOWRATE PREDICTION (LINEAR INTERPOLATION)")
    print("Assumption: Physical flow sensor (46-FIC-103A1.A) is uncalibrated")
    print("=" * 75)

    estimator = ValveFlowEstimator(cv_csv_path)

    print(f"\n1. Reading operational data from InfluxDB '{database}' (measurement: {table_name})...")
    df = query_dataframe(
        f'''
        SELECT
            "{TARGET_TAG}" AS sensor_flow_actual,
            "{VALVE_OPENING_TAG}" AS valve_opening
        FROM "{table_name}"
        ORDER BY time ASC
        ''',
        database=database,
    )
    if df.empty:
        raise RuntimeError(f"No data found in InfluxDB '{database}.{table_name}'.")
    print(f"   Total records loaded: {len(df):,} rows")

    print("\n2. Computing Flowrate Predictions via Linear Interpolation (Opening -> Cv)...")
    df["estimated_cv"] = estimator.opening_to_cv(df["valve_opening"])
    df["predicted_flow"] = estimator.predict_flow(df["valve_opening"])
    df["sensor_deviation"] = df["sensor_flow_actual"] - df["predicted_flow"]

    print("\n3. Valve Characteristic Table & Linear Formula:")
    table_display = estimator.cv_df.copy()
    table_display["flow_formula (48.4467*Cv)"] = (table_display["cv"] * estimator.cv_to_flow_factor).round(2)
    print(table_display.to_string(index=False))

    print("\n4. Physical Sensor Drift Analysis vs Linear Valve Model (Opening > 0.5%):")
    drift_stats = estimator.analyze_sensor_drift(df["valve_opening"].values, df["sensor_flow_actual"].values)
    for k, v in drift_stats.items():
        print(f"   - {k:<48}: {v}")

    if save_to_db:
        from src.config import TIMESCALE_DATABASE, TIMESCALE_SCHEMA

        print(f"\n5. Writing predictions to TimescaleDB table: '{TIMESCALE_SCHEMA}.{OUTPUT_TABLE_LINEAR}'...")
        df_out = pd.DataFrame({
            "Timestamp": df["Timestamp"],
            "46-FIC-103A.Q_opening": df["valve_opening"],
            "estimated_cv_linear": df["estimated_cv"],
            "predicted_flow_linear": df["predicted_flow"],
            "measured_fuelgas": df["sensor_flow_actual"],
            "sensor_drift_deviation": df["sensor_deviation"],
        })
        write_timescale_dataframe(df_out, table_name=OUTPUT_TABLE_LINEAR, database=TIMESCALE_DATABASE, schema=TIMESCALE_SCHEMA)
        print(f"   Table '{TIMESCALE_SCHEMA}.{OUTPUT_TABLE_LINEAR}' updated successfully ({len(df_out):,} rows).")

    print("\n" + "=" * 75)
    print("LINEAR INTERPOLATION PREDICTION COMPLETED!")
    print("=" * 75)
    return df, estimator

