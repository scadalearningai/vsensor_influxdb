"""
Inference engine and Sensor Drift Detection for Decoupled Soft Sensor.
Purely relies on Membrane Unit & WHB Unit (10 physical features).
"""

import os
import sys
import argparse
import duckdb
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(CURRENT_DIR)
WORKSPACE_ROOT = os.path.dirname(ML_DIR)
VSENSOR_ROOT = os.path.dirname(WORKSPACE_ROOT)
for p in [ML_DIR, WORKSPACE_ROOT, VSENSOR_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from common.influx_client import query_dataframe, write_dataframe
from src.config import (
    INFLUX_DATABASE, HISTORIAN_TABLE, OUTPUT_MEASUREMENT, OUTPUT_DAILY_MEASUREMENT,
    DB_INPUT_PATH, DB_OUTPUT_PATH, MODEL_PATH, TARGET_TAG,
    DRIFT_REL_THRESHOLD, EWMA_ALPHA, MIN_ACTIVE_FLOW
)
from src.features import build_features, get_feature_columns

MODEL_DECOUPLED_PATH = MODEL_PATH

class SoftSensorEngine:
    def __init__(self, model_path: str = MODEL_DECOUPLED_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}. Run src/train.py first.")
        
        artifact = joblib.load(model_path)
        self.model = artifact['model']
        self.feature_cols = artifact['feature_cols']
        self.metrics = artifact.get('metrics', {})
        self.ewma_drift = 0.0

    def predict_dataframe(self, df_raw: pd.DataFrame, disable_ewma: bool = False) -> pd.DataFrame:
        """
        Runs batch prediction with physics constraints (whb1_on gating)
        and sensor drift detection on raw data stream.
        """
        df = build_features(df_raw, is_training=False)
        X = df[self.feature_cols]

        # 1. Raw Machine Learning Prediction (y_hat)
        df['y_hat'] = self.model.predict(X)

        # 2. Physics-Constrained Flow (F_hat)
        # If boiler is OFF (whb1_on == 0), clamp to 0.0. Otherwise, clamp negative predictions to 0.0.
        df['F_hat'] = np.where(df['whb1_on'] == 0, 0.0, np.maximum(0.0, df['y_hat']))

        # 3. Sensor Drift Evaluation (if actual target 46-FQI-106A exists)
        if TARGET_TAG in df.columns:
            actual = df[TARGET_TAG]
            df['deviation_abs'] = actual - df['F_hat']
            
            # Relative deviation calculated only when burner is active
            active_mask = (df['whb1_on'] == 1) & (df['F_hat'] >= MIN_ACTIVE_FLOW)
            rel_dev = np.where(
                active_mask,
                np.abs(df['deviation_abs']) / np.maximum(MIN_ACTIVE_FLOW, df['F_hat']),
                0.0
            )
            df['deviation_rel'] = rel_dev
            
            if disable_ewma:
                df['ewma_drift'] = np.nan
                df['drift_alarm'] = (
                    (df['deviation_rel'] > DRIFT_REL_THRESHOLD) & (df['whb1_on'] == 1)
                ).astype(int)
            else:
                # EWMA smoothed relative drift
                df['ewma_drift'] = pd.Series(rel_dev, index=df.index).ewm(alpha=EWMA_ALPHA, adjust=False).mean()
                
                # Drift alarm triggered if persistent deviation exceeds threshold
                df['drift_alarm'] = (
                    (df['ewma_drift'] > DRIFT_REL_THRESHOLD) & (df['whb1_on'] == 1)
                ).astype(int)

        return df

    def predict_single(self, row_dict: Dict[str, Any], disable_ewma: bool = False) -> Dict[str, Any]:
        """
        Runs single-step prediction for real-time DCS/SCADA integration.
        """
        df_single = pd.DataFrame([row_dict])
        res_df = self.predict_dataframe(df_single, disable_ewma=disable_ewma)
        row = res_df.iloc[0]

        return {
            'y_hat': float(row['y_hat']),
            'F_hat': float(row['F_hat']),
            'whb1_on': int(row['whb1_on']),
            'deviation_abs': float(row.get('deviation_abs', 0.0)),
            'deviation_rel': float(row.get('deviation_rel', 0.0)),
            'ewma_drift': float(row.get('ewma_drift', 0.0)),
            'drift_alarm': int(row.get('drift_alarm', 0))
        }

# Backward compatibility alias
NonAmineSoftSensorEngine = SoftSensorEngine

def run_inference(
    lookback_minutes: Optional[int] = None,
    write_minutes: Optional[int] = None,
    skip_daily: bool = False,
    delay_minutes: int = 0,
    end_time: Optional[str] = None,
    disable_ewma: bool = False,
    single_point: bool = False,
):
    print(f"Loading trained soft sensor model from {MODEL_PATH}...")
    engine = SoftSensorEngine(MODEL_PATH)

    if single_point:
        if end_time is not None:
            t_end = pd.to_datetime(end_time).strftime("%Y-%m-%dT%H:%M:%SZ")
            query_str = (
                f'SELECT * FROM "{HISTORIAN_TABLE}" '
                f"WHERE time <= '{t_end}' "
                f'ORDER BY time DESC LIMIT 1'
            )
        elif delay_minutes > 0:
            query_str = (
                f'SELECT * FROM "{HISTORIAN_TABLE}" '
                f'WHERE time <= now() - {int(delay_minutes)}m '
                f'ORDER BY time DESC LIMIT 1'
            )
        else:
            query_str = f'SELECT * FROM "{HISTORIAN_TABLE}" ORDER BY time DESC LIMIT 1'
        print(f"Reading 1 latest point from InfluxDB measurement '{HISTORIAN_TABLE}' (DB: '{INFLUX_DATABASE}')...")
    elif end_time is not None:
        end_dt = pd.to_datetime(end_time)
        lb = lookback_minutes if lookback_minutes is not None else 180
        start_dt = end_dt - pd.Timedelta(minutes=int(lb))
        t_start = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        t_end = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        query_str = (
            f'SELECT * FROM "{HISTORIAN_TABLE}" '
            f"WHERE time >= '{t_start}' AND time <= '{t_end}' "
            f'ORDER BY time ASC'
        )
        print(
            f"Reading data from '{HISTORIAN_TABLE}' (DB: '{INFLUX_DATABASE}') "
            f"for window {t_start} to {t_end}..."
        )
    elif lookback_minutes is not None:
        total_back = int(lookback_minutes) + int(delay_minutes)
        if delay_minutes > 0:
            time_filter = f'WHERE time >= now() - {total_back}m AND time <= now() - {int(delay_minutes)}m'
        else:
            time_filter = f'WHERE time >= now() - {total_back}m'
        query_str = (
            f'SELECT * FROM "{HISTORIAN_TABLE}" '
            f'{time_filter} '
            f'ORDER BY time ASC'
        )
        print(
            f"Reading {lookback_minutes}m data (delay {delay_minutes}m) from InfluxDB measurement "
            f"'{HISTORIAN_TABLE}' (DB: '{INFLUX_DATABASE}')..."
        )
    else:
        query_str = f'SELECT * FROM "{HISTORIAN_TABLE}" ORDER BY time ASC'
        print(f"Reading raw data from InfluxDB measurement '{HISTORIAN_TABLE}' (DB: '{INFLUX_DATABASE}')...")

    df_raw = query_dataframe(query_str, database=INFLUX_DATABASE)
    if df_raw.empty:
        print("No records found in InfluxDB for the requested window. Skipping inference.")
        return
    print(f"Loaded {len(df_raw):,} records from InfluxDB.")

    print(f"Running soft sensor inference & drift pipeline for {len(df_raw):,} records (disable_ewma={disable_ewma})...")
    df_pred = engine.predict_dataframe(df_raw, disable_ewma=disable_ewma)

    measured = df_pred[TARGET_TAG].fillna(0.0).values
    F_hat = df_pred['F_hat'].values
    whb1_on = df_pred['whb1_on'].values
    deviation = df_pred.get('deviation_abs', measured - F_hat)
    rel_dev = df_pred.get('deviation_rel', np.zeros_like(F_hat))
    rel_dev_ewma = df_pred.get('ewma_drift', np.zeros_like(F_hat))
    drift_alarm = df_pred.get('drift_alarm', np.zeros_like(F_hat, dtype=int))

    df_output_full = pd.DataFrame({
        'Timestamp': df_pred['Timestamp'],
        'measured_106A': measured,
        'whb1_on': whb1_on,
        'feed_total': df_pred['feed_total'],
        'P_total': df_pred['P_total'],
        'y_hat': df_pred['y_hat'],
        'F_hat': F_hat,
        'deviation': deviation,
        'rel_deviation': rel_dev,
        'rel_dev_ewma': rel_dev_ewma,
        'drift_alarm': drift_alarm,
        'trusted': (drift_alarm == 0).astype(int),
        'suspect': drift_alarm
    })

    # Optional: filter output to only the newest points (e.g., for 5-minute cron)
    if write_minutes is not None and not df_output_full.empty:
        max_ts = pd.to_datetime(df_output_full['Timestamp']).max()
        cutoff_ts = max_ts - pd.Timedelta(minutes=int(write_minutes))
        df_to_write = df_output_full[pd.to_datetime(df_output_full['Timestamp']) >= cutoff_ts].copy()
    else:
        df_to_write = df_output_full

    if skip_daily:
        print(f"Writing {len(df_to_write):,} latest points to InfluxDB measurement '{OUTPUT_MEASUREMENT}'...")
        write_dataframe(df_to_write, measurement=OUTPUT_MEASUREMENT, database=INFLUX_DATABASE)
        print(f"InfluxDB write complete: {len(df_to_write):,} points written (daily aggregation skipped).")
        return

    # Daily aggregation
    df_output_full['Date'] = pd.to_datetime(df_output_full['Timestamp']).dt.date
    daily_full = df_output_full.groupby('Date').agg(
        measured_total_lbs=('measured_106A', lambda s: np.sum(s) / 60.0),
        softsensor_total_lbs=('F_hat', lambda s: np.sum(s) / 60.0),
        drift_alarm_frac=('drift_alarm', 'mean')
    ).reset_index()

    denom_daily = np.where(daily_full['softsensor_total_lbs'] > 1000.0, daily_full['softsensor_total_lbs'], np.nan)
    daily_full['diff_%'] = np.round(((daily_full['measured_total_lbs'] - daily_full['softsensor_total_lbs']) / denom_daily) * 100.0, 2)
    daily_full['diff_%'] = daily_full['diff_%'].fillna(0.0)

    print(f"Writing outputs to InfluxDB measurements: '{OUTPUT_MEASUREMENT}' and '{OUTPUT_DAILY_MEASUREMENT}'...")
    write_dataframe(df_to_write, measurement=OUTPUT_MEASUREMENT, database=INFLUX_DATABASE)
    write_dataframe(daily_full, measurement=OUTPUT_DAILY_MEASUREMENT, database=INFLUX_DATABASE, time_column="Date")
    print(f"InfluxDB write complete: {len(df_to_write):,} full points, {len(daily_full):,} daily points written.")


def main():
    parser = argparse.ArgumentParser(description="Run Soft Sensor (46-FQI-106A) inference.")
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Run incremental inference on latest data window (default: 180m lookback, 5m write, skip daily).",
    )
    parser.add_argument(
        "--lookback-minutes",
        type=int,
        default=None,
        help="Query only the last N minutes from InfluxDB (e.g., 180 for EWMA warmup).",
    )
    parser.add_argument(
        "--write-minutes",
        type=int,
        default=None,
        help="Write only points within the last N minutes of the latest timestamp.",
    )
    parser.add_argument(
        "--skip-daily",
        action="store_true",
        help="Skip writing softsensor_permeate_daily (recommended for incremental cron runs).",
    )
    parser.add_argument(
        "--delay-minutes",
        type=int,
        default=0,
        help="Shift target end time back by N minutes from now() (e.g., 30 when cron at 05:30 targets 05:00).",
    )
    parser.add_argument(
        "--end-time",
        type=str,
        default=None,
        help="Explicit target end timestamp (e.g., '2026-11-25 05:00:00') instead of now().",
    )
    parser.add_argument(
        "--disable-ewma",
        action="store_true",
        help="Disable EWMA drift smoothing (uses instantaneous relative deviation for drift_alarm).",
    )
    parser.add_argument(
        "--single-point",
        action="store_true",
        help="Query only 1 latest point (ORDER BY time DESC LIMIT 1); implies --disable-ewma and --skip-daily.",
    )
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


if __name__ == '__main__':
    main()
