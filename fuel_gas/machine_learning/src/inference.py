"""
Inference Engine & Sensor Drift Detection for Fuel Gas Soft Sensor (46-FIC-103A1.A).

Supports:
1. Real-time single-row stateless inference (`predict_single`).
2. Full-historian batch inference & EWMA drift monitoring exported to DuckDB.
"""

import os
import sys
import argparse
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(ML_DIR)
VSENSOR_ROOT = os.path.dirname(PROJECT_ROOT)
for p in (ML_DIR, PROJECT_ROOT, VSENSOR_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.influx_client import query_dataframe
from common.timescale_client import write_dataframe as write_timescale_dataframe
from src.config import (
    INFLUX_DATABASE,
    TIMESCALE_DATABASE,
    TIMESCALE_SCHEMA,
    HISTORIAN_TABLE,
    OUTPUT_MEASUREMENT,
    OUTPUT_DAILY_MEASUREMENT,
    DB_HISTORIAN_PATH,
    DB_OUTPUT_PATH,
    DRIFT_REL_THRESHOLD,
    EWMA_ALPHA,
    MIN_ACTIVE_FLOW,
    MODEL_PATH,
    REQUIRED_RAW_TAGS,
    TARGET_TAG,
)
from src.features import build_features


class FuelGasSoftSensorEngine:
    """Stateless soft sensor inference and EWMA sensor drift detector."""

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found: {model_path}. Run src/train.py first.")
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.model_name = artifact.get("model_name", "Unknown")
        self.feature_cols = artifact["feature_cols"]
        self.metrics = artifact.get("winner_metrics", {})
        self._patch_imputers(self.model)

    @staticmethod
    def _patch_imputers(obj):
        """Patches SimpleImputer objects for sklearn >= 1.6 compatibility."""
        if hasattr(obj, "fitted_bases_") and isinstance(obj.fitted_bases_, dict):
            for base in obj.fitted_bases_.values():
                FuelGasSoftSensorEngine._patch_imputers(base)
        if hasattr(obj, "estimator"):
            FuelGasSoftSensorEngine._patch_imputers(obj.estimator)
        if hasattr(obj, "named_steps"):
            for step in obj.named_steps.values():
                FuelGasSoftSensorEngine._patch_imputers(step)
        if hasattr(obj, "_fit_dtype") and not hasattr(obj, "_fill_dtype"):
            obj._fill_dtype = obj._fit_dtype

    def predict_dataframe(self, df_raw: pd.DataFrame, disable_ewma: bool = False) -> pd.DataFrame:
        """Runs stateless batch prediction with physics gating and drift monitoring."""
        df = build_features(df_raw, is_training=False)
        active_mask = df["fuel_gas_active"].to_numpy() == 1
        y_hat = np.zeros(len(df), dtype=float)
        if np.any(active_mask):
            y_hat[active_mask] = self.model.predict(df.loc[active_mask, self.feature_cols])

        df["y_hat"] = y_hat
        df["F_hat"] = np.where(active_mask, np.maximum(0.0, y_hat), 0.0)

        if TARGET_TAG in df.columns:
            actual = pd.to_numeric(df[TARGET_TAG], errors="coerce").to_numpy(dtype=float)
            f_hat = df["F_hat"].to_numpy(dtype=float)
            active = df["fuel_gas_active"].to_numpy(dtype=int)

            df["deviation_abs"] = actual - f_hat
            eval_mask = (active == 1) & (f_hat >= MIN_ACTIVE_FLOW) & (~np.isnan(actual))
            rel_dev = np.where(
                eval_mask,
                np.abs(df["deviation_abs"]) / np.maximum(MIN_ACTIVE_FLOW, f_hat),
                0.0,
            )
            df["deviation_rel"] = rel_dev
            if disable_ewma:
                df["ewma_drift"] = np.nan
                df["drift_alarm"] = ((df["deviation_rel"] > DRIFT_REL_THRESHOLD) & (active == 1)).astype(int)
            else:
                df["ewma_drift"] = pd.Series(rel_dev, index=df.index).ewm(alpha=EWMA_ALPHA, adjust=False).mean()
                df["drift_alarm"] = ((df["ewma_drift"] > DRIFT_REL_THRESHOLD) & (active == 1)).astype(int)

        return df

    def predict_single(self, row_dict: Dict[str, Any], disable_ewma: bool = False) -> Dict[str, Any]:
        """Runs single-step prediction for real-time DCS/SCADA integration."""
        res_df = self.predict_dataframe(pd.DataFrame([row_dict]), disable_ewma=disable_ewma)
        row = res_df.iloc[0]
        return {
            "y_hat": float(row["y_hat"]),
            "F_hat": float(row["F_hat"]),
            "whb1_on": int(row["whb1_on"]),
            "valve_open": int(row["valve_open"]),
            "fuel_gas_active": int(row["fuel_gas_active"]),
            "deviation_abs": float(row.get("deviation_abs", 0.0)),
            "deviation_rel": float(row.get("deviation_rel", 0.0)),
            "ewma_drift": float(row.get("ewma_drift", 0.0)),
            "drift_alarm": int(row.get("drift_alarm", 0)),
        }


def run_inference(
    lookback_minutes: Optional[int] = None,
    write_minutes: Optional[int] = None,
    skip_daily: bool = False,
    delay_minutes: int = 0,
    end_time: Optional[str] = None,
    disable_ewma: bool = False,
    single_point: bool = False,
) -> None:
    print("=" * 75)
    print("FUEL GAS SOFT SENSOR - INFERENCE & DRIFT PIPELINE")
    print("=" * 75)
    engine = FuelGasSoftSensorEngine(MODEL_PATH)
    print(f"Loaded model '{engine.model_name}' from: {MODEL_PATH}")

    select_cols = [TARGET_TAG] + REQUIRED_RAW_TAGS
    quoted_cols = ", ".join([f'"{t}"' for t in select_cols])

    if single_point:
        if end_time is not None:
            t_end = pd.to_datetime(end_time).strftime("%Y-%m-%dT%H:%M:%SZ")
            query_str = (
                f'SELECT {quoted_cols} FROM "{HISTORIAN_TABLE}" '
                f"WHERE time <= '{t_end}' ORDER BY time DESC LIMIT 1"
            )
        elif delay_minutes > 0:
            query_str = (
                f'SELECT {quoted_cols} FROM "{HISTORIAN_TABLE}" '
                f'WHERE time <= now() - {int(delay_minutes)}m ORDER BY time DESC LIMIT 1'
            )
        else:
            query_str = f'SELECT {quoted_cols} FROM "{HISTORIAN_TABLE}" ORDER BY time DESC LIMIT 1'
        print(f"Reading 1 latest point from InfluxDB measurement '{HISTORIAN_TABLE}' (DB: '{INFLUX_DATABASE}')...")
    elif end_time is not None:
        end_dt = pd.to_datetime(end_time)
        lb = lookback_minutes if lookback_minutes is not None else 180
        start_dt = end_dt - pd.Timedelta(minutes=int(lb))
        t_start = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        t_end = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        query_str = (
            f'SELECT {quoted_cols} FROM "{HISTORIAN_TABLE}" '
            f"WHERE time >= '{t_start}' AND time <= '{t_end}' ORDER BY time ASC"
        )
        print(f"Reading data from '{HISTORIAN_TABLE}' (DB: '{INFLUX_DATABASE}') for window {t_start} to {t_end}...")
    elif lookback_minutes is not None:
        total_back = int(lookback_minutes) + int(delay_minutes)
        if delay_minutes > 0:
            time_filter = f'WHERE time >= now() - {total_back}m AND time <= now() - {int(delay_minutes)}m'
        else:
            time_filter = f'WHERE time >= now() - {total_back}m'
        query_str = f'SELECT {quoted_cols} FROM "{HISTORIAN_TABLE}" {time_filter} ORDER BY time ASC'
        print(f"Reading {lookback_minutes}m data (delay {delay_minutes}m) from '{HISTORIAN_TABLE}' (DB: '{INFLUX_DATABASE}')...")
    else:
        query_str = f'SELECT {quoted_cols} FROM "{HISTORIAN_TABLE}" ORDER BY time ASC'
        print(f"Reading input data from InfluxDB measurement '{HISTORIAN_TABLE}' (DB: '{INFLUX_DATABASE}')...")

    df_raw = query_dataframe(query_str, database=INFLUX_DATABASE)
    if df_raw.empty:
        print("No records found in InfluxDB for the requested window. Skipping inference.")
        return
    print(f"Loaded {len(df_raw):,} historian records from InfluxDB.")

    df_pred = engine.predict_dataframe(df_raw, disable_ewma=disable_ewma)
    measured = pd.to_numeric(df_pred[TARGET_TAG], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    f_hat = df_pred["F_hat"].to_numpy(dtype=float)

    df_output_full = pd.DataFrame({
        "Timestamp": df_pred["Timestamp"],
        "measured_fuelgas": measured,
        "valve_opening_pct": df_pred["46-FIC-103A1.Q"],
        "whb1_on": df_pred["whb1_on"],
        "valve_open": df_pred["valve_open"],
        "fuel_gas_active": df_pred["fuel_gas_active"],
        "y_hat": df_pred["y_hat"],
        "F_hat": f_hat,
        "deviation": df_pred["deviation_abs"].fillna(0.0),
        "rel_deviation": df_pred["deviation_rel"],
        "rel_dev_ewma": df_pred["ewma_drift"],
        "drift_alarm": df_pred["drift_alarm"],
        "trusted": (df_pred["drift_alarm"] == 0).astype(int),
        "suspect": df_pred["drift_alarm"],
    })

    if write_minutes is not None and not df_output_full.empty:
        max_ts = pd.to_datetime(df_output_full["Timestamp"]).max()
        cutoff_ts = max_ts - pd.Timedelta(minutes=int(write_minutes))
        df_to_write = df_output_full[pd.to_datetime(df_output_full["Timestamp"]) >= cutoff_ts].copy()
    else:
        df_to_write = df_output_full

    if skip_daily:
        print(f"Writing {len(df_to_write):,} latest points to TimescaleDB table '{TIMESCALE_SCHEMA}.{OUTPUT_MEASUREMENT}'...")
        write_timescale_dataframe(df_to_write, table_name=OUTPUT_MEASUREMENT, database=TIMESCALE_DATABASE, schema=TIMESCALE_SCHEMA)
        print(f"TimescaleDB export complete: {len(df_to_write):,} points written (daily aggregation skipped).")
        return

    ts_diffs = pd.to_datetime(df_output_full["Timestamp"]).diff().dt.total_seconds().dropna()
    dt_minutes = float(ts_diffs.median() / 60.0) if not ts_diffs.empty else 5.0
    dt_hours = dt_minutes / 60.0
    print(f"Detected historian sampling interval: {dt_minutes:.2f} minutes (mass integration factor: {dt_hours:.5f} hr/row)")

    df_output_full["Date"] = pd.to_datetime(df_output_full["Timestamp"]).dt.date
    daily_full = (
        df_output_full.groupby("Date")
        .agg(
            measured_total_lbs=("measured_fuelgas", lambda s: float(np.sum(s) * dt_hours)),
            softsensor_total_lbs=("F_hat", lambda s: float(np.sum(s) * dt_hours)),
            active_minutes=("fuel_gas_active", lambda s: float(np.sum(s) * dt_minutes)),
            drift_alarm_frac=("drift_alarm", "mean"),
        )
        .reset_index()
    )
    denom = np.where(daily_full["softsensor_total_lbs"] > 500.0, daily_full["softsensor_total_lbs"], np.nan)
    daily_full["diff_pct"] = np.round(
        ((daily_full["measured_total_lbs"] - daily_full["softsensor_total_lbs"]) / denom) * 100.0, 2
    )
    daily_full["diff_pct"] = daily_full["diff_pct"].fillna(0.0)
    df_output_full = df_output_full.drop(columns=["Date"])

    print(f"Writing batch inference outputs to TimescaleDB tables: '{TIMESCALE_SCHEMA}.{OUTPUT_MEASUREMENT}' & '{TIMESCALE_SCHEMA}.{OUTPUT_DAILY_MEASUREMENT}'")
    write_timescale_dataframe(df_to_write, table_name=OUTPUT_MEASUREMENT, database=TIMESCALE_DATABASE, schema=TIMESCALE_SCHEMA)
    write_timescale_dataframe(daily_full, table_name=OUTPUT_DAILY_MEASUREMENT, database=TIMESCALE_DATABASE, schema=TIMESCALE_SCHEMA, time_column="Date")
    print(f"TimescaleDB export complete: {len(df_to_write):,} full points and {len(daily_full):,} daily points written.")

    if not df_output_full.loc[df_output_full["fuel_gas_active"] == 0].empty:
        shutdown_max = float(df_output_full.loc[df_output_full["fuel_gas_active"] == 0, "F_hat"].max())
        print(f"Physics gating check (max F_hat when inactive): {shutdown_max:.6f} LBS/HR")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Fuel Gas Soft Sensor (46-FIC-103A1.A) inference.")
    parser.add_argument("--latest", action="store_true", help="Run incremental inference on latest data window (180m lookback, 5m write, skip daily).")
    parser.add_argument("--lookback-minutes", type=int, default=None, help="Query only the last N minutes from InfluxDB.")
    parser.add_argument("--write-minutes", type=int, default=None, help="Write only points within the last N minutes of the latest timestamp.")
    parser.add_argument("--skip-daily", action="store_true", help="Skip writing softsensor_fuelgas_daily.")
    parser.add_argument("--delay-minutes", type=int, default=0, help="Shift target end time back by N minutes from now().")
    parser.add_argument("--end-time", type=str, default=None, help="Explicit target end timestamp (e.g., '2026-11-25 05:00:00').")
    parser.add_argument("--disable-ewma", action="store_true", help="Disable EWMA drift smoothing.")
    parser.add_argument("--single-point", action="store_true", help="Query only 1 latest point (ORDER BY time DESC LIMIT 1); implies --disable-ewma and --skip-daily.")
    args = parser.parse_args()

    single_point = args.single_point
    disable_ewma = args.disable_ewma or single_point
    use_incremental = args.latest or (args.end_time is not None) or (args.delay_minutes > 0) or single_point
    lookback = None if single_point else (
        args.lookback_minutes if args.lookback_minutes is not None else (180 if use_incremental else None)
    )
    write_win = None if single_point else (
        args.write_minutes if args.write_minutes is not None else (5 if use_incremental else None)
    )
    skip_daily = args.skip_daily or use_incremental

    run_inference(
        lookback_minutes=lookback,
        write_minutes=write_win,
        skip_daily=skip_daily,
        delay_minutes=args.delay_minutes,
        end_time=args.end_time,
        disable_ewma=disable_ewma,
        single_point=single_point,
    )


if __name__ == "__main__":
    main()

