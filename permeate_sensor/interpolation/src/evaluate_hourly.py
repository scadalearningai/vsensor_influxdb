"""
Hourly Evaluation Script for Permeate Flow Two-Stage Interpolation.
Evaluates estimated permeate flow against actual sensor readings from DuckDB.
Default date: 2026-09-09 (00:00 - 23:00), dynamically supports any date.
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from .config import (
        DEFAULT_PERMEATE_CSV,
        FIGURES_DIR,
        HISTORIAN_TABLE,
        INFLUX_DATABASE,
        VSENSOR_ROOT,
    )
    from .estimator import PermeateFlowEstimator
except ImportError:
    from config import (
        DEFAULT_PERMEATE_CSV,
        FIGURES_DIR,
        HISTORIAN_TABLE,
        INFLUX_DATABASE,
        VSENSOR_ROOT,
    )
    from estimator import PermeateFlowEstimator

if VSENSOR_ROOT not in sys.path:
    sys.path.insert(0, VSENSOR_ROOT)

from common.influx_client import query_interval_tags


def fetch_hourly_data(
    database: str,
    target_date: str,
    mode: str = "mean"
) -> pd.DataFrame:
    """
    Fetch hourly sensor data (00:00 to 23:00) from InfluxDB for a specific date.

    Args:
        database: InfluxDB database name (default: whb_medco)
        target_date: Date string in 'YYYY-MM-DD' format
        mode: 'mean' (hourly average) or 'snapshot' (exact :00 minute points)
    """
    start_ts = f"{target_date} 00:00:00"
    end_ts = f"{target_date} 23:59:59"

    df_raw = query_interval_tags(
        tags=["46-FIC-106A.Q", "46-FV-106A", "46-FIC-106A.A"],
        start_dt=start_ts,
        end_dt=end_ts,
        measurement=HISTORIAN_TABLE,
        database=database,
    )

    if df_raw.empty:
        raise ValueError(f"No data found in InfluxDB '{database}.{HISTORIAN_TABLE}' for date {target_date}.")

    df_raw["Timestamp"] = pd.to_datetime(df_raw["Timestamp"])
    if "46-FIC-106A.Q" in df_raw.columns and "46-FV-106A" in df_raw.columns:
        df_raw["fv_opening"] = df_raw["46-FIC-106A.Q"].combine_first(df_raw["46-FV-106A"])
    elif "46-FIC-106A.Q" in df_raw.columns:
        df_raw["fv_opening"] = df_raw["46-FIC-106A.Q"]
    else:
        df_raw["fv_opening"] = df_raw["46-FV-106A"]

    df_raw["fic_actual"] = df_raw["46-FIC-106A.A"]

    if mode == "mean":
        df_raw["hour_ts"] = df_raw["Timestamp"].dt.floor("h")
        df = (
            df_raw.groupby("hour_ts", as_index=False)
            .agg(
                fv_opening=("fv_opening", "mean"),
                fic_actual=("fic_actual", "mean"),
                points_count=("fv_opening", "count"),
            )
            .sort_values("hour_ts")
            .reset_index(drop=True)
        )
    elif mode == "snapshot":
        df = df_raw[df_raw["Timestamp"].dt.minute == 0].copy()
        df = df.rename(columns={"Timestamp": "hour_ts"})
        df["points_count"] = 1
        df = df[["hour_ts", "fv_opening", "fic_actual", "points_count"]].sort_values("hour_ts").reset_index(drop=True)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'mean' or 'snapshot'.")

    if df.empty:
        raise ValueError(f"No data found in InfluxDB for date {target_date} (mode={mode}).")

    return df


def evaluate_date(
    target_date: str = "2026-09-09",
    mode: str = "mean",
    database: str = None,
    csv_path: str = None,
    save_plot: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run 24-hour evaluation for a specific date.

    Returns:
        (df_results, metrics_dict)
    """
    database = database or INFLUX_DATABASE
    csv_path = csv_path or DEFAULT_PERMEATE_CSV

    # 1. Inisialisasi Estimator
    estimator = PermeateFlowEstimator(csv_path)

    # 2. Ambil data historis
    df = fetch_hourly_data(database, target_date, mode=mode)

    # 3. Prediksi menggunakan metode 2-tahap
    calculated_cv, predicted_flow = estimator.predict(df["fv_opening"].to_numpy())

    df["calc_cv"] = calculated_cv
    df["pred_flow"] = predicted_flow
    df["abs_error"] = np.abs(df["fic_actual"] - df["pred_flow"])
    
    # Error persentase relatif terhadap aktual (hindari pembagian 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["pct_error"] = np.where(
            df["fic_actual"] > 0,
            (df["abs_error"] / df["fic_actual"]) * 100.0,
            0.0
        )

    # 4. Hitung Metrik Evaluasi
    valid_mask = df["fic_actual"].notnull() & df["pred_flow"].notnull()
    act = df.loc[valid_mask, "fic_actual"].to_numpy()
    pred = df.loc[valid_mask, "pred_flow"].to_numpy()
    abs_err = df.loc[valid_mask, "abs_error"].to_numpy()

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((act - pred) ** 2)))
    sum_act = float(np.sum(act))
    wmape = float((np.sum(abs_err) / sum_act) * 100.0) if sum_act > 0 else 0.0

    metrics = {
        "date": target_date,
        "mode": mode,
        "linear_eq": estimator.linear_equation_str,
        "total_hours": len(df),
        "avg_opening": float(df["fv_opening"].mean()),
        "avg_actual_flow": float(df["fic_actual"].mean()),
        "avg_pred_flow": float(df["pred_flow"].mean()),
        "mae": mae,
        "rmse": rmse,
        "wmape_pct": wmape,
        "max_abs_error": float(abs_err.max()),
        "min_abs_error": float(abs_err.min())
    }

    # 5. Visualisasi jika diminta
    if save_plot:
        plot_path = os.path.join(FIGURES_DIR, f"hourly_eval_{target_date}_{mode}.png")
        generate_plot(df, metrics, plot_path)
        metrics["plot_path"] = plot_path

    return df, metrics


def generate_plot(df: pd.DataFrame, metrics: Dict[str, Any], output_path: str) -> None:
    """Generate 24-hour comparison plot (Actual vs Predicted Flow & Valve Opening)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    hours = [pd.to_datetime(ts).strftime("%H:%M") for ts in df["hour_ts"]]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left axis: Flowrate
    color_act = "#1f77b4"
    color_pred = "#d62728"
    
    ax1.plot(hours, df["fic_actual"], marker="o", color=color_act, linewidth=2, label="Actual Sensor (46-FIC-106A.A)")
    ax1.plot(hours, df["pred_flow"], marker="s", linestyle="--", color=color_pred, linewidth=2, label="Two-Stage Estimate (Virtual Sensor)")
    ax1.set_xlabel("Hour (Operating Time)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Permeate Flowrate", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.tick_params(axis="x", rotation=45)

    # Right axis: Valve Opening (%)
    ax2 = ax1.twinx()
    color_valve = "#2ca02c"
    ax2.plot(hours, df["fv_opening"], marker="^", color=color_valve, linestyle=":", alpha=0.8, label="Valve Opening (46-FV-106A)")
    ax2.set_ylabel("Valve Opening (%)", fontsize=11, color=color_valve, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_valve)
    ax2.set_ylim(0, 100)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.9)

    mode_title = "Hourly Mean" if metrics["mode"] == "mean" else "Minute :00 Snapshot"
    plt.title(
        f"Two-Stage Permeate Flow Evaluation: {metrics['date']} ({mode_title})\n"
        f"MAE: {metrics['mae']:.2f} | RMSE: {metrics['rmse']:.2f} | WMAPE: {metrics['wmape_pct']:.2f}%",
        fontsize=12,
        fontweight="bold"
    )

    fig.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Hourly Two-Stage Permeate Flow Evaluation from InfluxDB")
    parser.add_argument("--date", type=str, default="2026-09-09", help="Target evaluation date (YYYY-MM-DD)")
    parser.add_argument("--mode", type=str, choices=["mean", "snapshot"], default="mean", help="Hourly aggregation mode")
    parser.add_argument("--plot", action="store_true", default=True, help="Save visualization plot")
    args = parser.parse_args()

    print("=" * 80)
    print("TWO-STAGE METHOD EVALUATION (Opening -> Cv via Interp | Cv -> Flow via Linear)")
    print(f"Target Date      : {args.date}")
    print(f"Aggregation Mode : {args.mode.upper()}")
    print("=" * 80)

    df, metrics = evaluate_date(target_date=args.date, mode=args.mode, save_plot=args.plot)

    print(f"Stage 2 Equation: {metrics['linear_eq']}")

    display_df = pd.DataFrame({
        "Hour": [pd.to_datetime(ts).strftime("%H:%M") for ts in df["hour_ts"]],
        "Opening (%)": df["fv_opening"].round(2),
        "Calc Cv": df["calc_cv"].round(1),
        "Pred Flow": df["pred_flow"].round(1),
        "Actual Flow": df["fic_actual"].round(1),
        "Abs Error": df["abs_error"].round(1),
        "Error (%)": df["pct_error"].round(2)
    })

    print("\n[24-Hour Evaluation Table (00:00 to 23:00)]")
    print(display_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("[Daily Performance Metrics Summary]")
    print(f"  Date                     : {metrics['date']}")
    print(f"  Mean Valve Opening       : {metrics['avg_opening']:.2f} %")
    print(f"  Mean Actual Flow         : {metrics['avg_actual_flow']:.2f}")
    print(f"  Mean Predicted Flow      : {metrics['avg_pred_flow']:.2f}")
    print(f"  MAE (Mean Absolute Error): {metrics['mae']:.2f}")
    print(f"  RMSE                     : {metrics['rmse']:.2f}")
    print(f"  WMAPE                    : {metrics['wmape_pct']:.2f} %")
    print(f"  Max Absolute Error       : {metrics['max_abs_error']:.2f}")
    print(f"  Min Absolute Error       : {metrics['min_abs_error']:.2f}")
    if "plot_path" in metrics:
        print(f"  Evaluation Plot Saved    : {metrics['plot_path']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
