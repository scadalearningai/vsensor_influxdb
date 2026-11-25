"""
Plotting script for the Fuel Gas Valve Linear Interpolation model.
Generates high-resolution 300 DPI figures into interpolation/reports/figures/.
"""

import os
import sys
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.dirname(SCRIPT_DIR)
if MODULE_ROOT not in sys.path:
    sys.path.insert(0, MODULE_ROOT)

from src.config import DEFAULT_CV_PATH, DEFAULT_DB_PATH, FIGURES_DIR, LOCAL_TAG_DB_PATH, OUTPUT_TABLE_LINEAR
from src.estimator import ValveFlowEstimator


def create_valve_characteristic_plot(
    estimator: ValveFlowEstimator,
    output_img: str = None,
) -> str:
    """
    Plot 3 valve characteristic curves (in English):
    1. Valve Opening (%) vs Cv
    2. Valve Opening (%) vs Predicted Flow Rate
    3. Cv vs Predicted Flow Rate
    """
    if output_img is None:
        output_img = os.path.join(FIGURES_DIR, "plot_valve_curve.png")
    os.makedirs(os.path.dirname(os.path.abspath(output_img)), exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5.5), dpi=300)

    csv_df = estimator.cv_df
    x_fine = np.linspace(0, 100, 300)
    cv_fine = estimator.opening_to_cv(x_fine)
    flow_pred_fine = estimator.predict_flow(x_fine)

    ax1.plot(x_fine, cv_fine, color="#1f77b4", linewidth=2.5, label="Piecewise Linear Fit")
    ax1.scatter(csv_df["opening"], csv_df["cv"], color="#d62728", s=55, zorder=5, label="cv_fuelgas.csv Points")
    ax1.set_title("Valve Opening (%) vs Cv", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Valve Opening (%)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Valve Flow Coefficient (Cv)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9.5)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(bottom=0)

    ax2.plot(x_fine, flow_pred_fine, color="#2ca02c", linewidth=2.5, label="Predicted Flow (Q = 48.4467 × Cv)")
    ax2.scatter(csv_df["opening"], csv_df["flowrate"], color="#2ca02c", s=45, zorder=5, label="Datasheet Flow Points")
    ax2.set_title("Valve Opening (%) vs Predicted Flow Rate", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("Valve Opening (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Predicted Flow Rate", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9.5)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(bottom=0)

    cv_range = np.linspace(0, float(csv_df["cv"].max()), 200)
    flow_from_cv = estimator.cv_to_flow_factor * cv_range
    ax3.plot(cv_range, flow_from_cv, color="#ff7f0e", linewidth=2.5, label=f"Linear Relation (Slope = {estimator.cv_to_flow_factor:.4f})")
    ax3.scatter(csv_df["cv"], csv_df["flowrate"], color="#d62728", s=50, zorder=5, label="Datasheet Reference Points")
    ax3.set_title("Valve Cv vs Predicted Flow Rate", fontsize=12, fontweight="bold", pad=12)
    ax3.set_xlabel("Valve Flow Coefficient (Cv)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Predicted Flow Rate", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9.5)
    ax3.set_xlim(0, float(csv_df["cv"].max()) * 1.05)
    ax3.set_ylim(bottom=0)

    fig.suptitle("Fuel Gas Control Valve 46-FIC-103A Physical Characteristics (Linear Model)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    plt.savefig(output_img, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Valve characteristic plot saved to: '{output_img}'")
    return output_img


def create_prediction_overview_plot(
    database: str = None,
    output_img: str = None,
) -> str:
    """Plot time-series predicted flow rate, valve opening, 2-week comparison, and drift distribution."""
    from src.config import TIMESCALE_DATABASE, TIMESCALE_SCHEMA, VSENSOR_ROOT
    if VSENSOR_ROOT not in sys.path:
        sys.path.insert(0, VSENSOR_ROOT)
    from common.timescale_client import query_dataframe

    target_db = database or TIMESCALE_DATABASE
    if output_img is None:
        output_img = os.path.join(FIGURES_DIR, "plot_prediction_overview.png")
    os.makedirs(os.path.dirname(os.path.abspath(output_img)), exist_ok=True)

    print(f"Reading prediction results from TimescaleDB: '{target_db}.{TIMESCALE_SCHEMA}.{OUTPUT_TABLE_LINEAR}'...")
    df = query_dataframe(
        f'''
        SELECT
            time,
            tag_46_fic_103a_q_opening AS opening,
            estimated_cv_linear AS cv,
            predicted_flow_linear AS predicted_flow,
            measured_fuelgas AS sensor_uncalibrated,
            sensor_drift_deviation AS drift_deviation
        FROM "{TIMESCALE_SCHEMA}"."{OUTPUT_TABLE_LINEAR}"
        ORDER BY time ASC
        ''',
        database=target_db,
    )
    if df.empty:
        raise ValueError(f"No data found in TimescaleDB '{target_db}.{TIMESCALE_SCHEMA}.{OUTPUT_TABLE_LINEAR}'.")

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(16, 11), dpi=300)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.6, 1.0, 1.4])

    ax_pred = fig.add_subplot(gs[0, :])
    ax_pred.plot(df["Timestamp"], df["predicted_flow"], color="#2ca02c", linewidth=0.8, alpha=0.9, label="Predicted Flow Rate (Valve Opening & Cv Model)")
    ax_pred.set_ylabel("Predicted Flow Rate", fontsize=11, fontweight="bold", color="#1b5e20")
    ax_pred.tick_params(axis="y", labelcolor="#1b5e20")
    ax_pred.set_title("Fuel Gas 46-FIC-103A Predicted Flow Rate (Linear Model: Opening & Cv)", fontsize=13, fontweight="bold", pad=10)
    ax_pred.grid(True, linestyle="--", alpha=0.5)
    ax_pred.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    ax_valve = fig.add_subplot(gs[1, :], sharex=ax_pred)
    ax_valve.plot(df["Timestamp"], df["opening"], color="#1f77b4", linewidth=0.7, alpha=0.85, label="Valve Opening (%) (46-FIC-103A.Q)")
    ax_valve.set_ylabel("Opening (%)", fontsize=11, fontweight="bold", color="#0d47a1")
    ax_valve.tick_params(axis="y", labelcolor="#0d47a1")
    ax_valve.set_xlabel("Timestamp", fontsize=11, fontweight="bold")
    ax_valve.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_valve.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax_pred.get_xticklabels(), visible=False)
    plt.setp(ax_valve.xaxis.get_majorticklabels(), rotation=25, ha="right")
    ax_valve.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    ax_valve.grid(True, linestyle="--", alpha=0.5)

    ax_comp = fig.add_subplot(gs[2, 0])
    sample_period = df[(df["Timestamp"] >= "2026-03-01") & (df["Timestamp"] <= "2026-03-15")]
    if sample_period.empty:
        sample_period = df.iloc[100000:120000]

    ax_comp.plot(sample_period["Timestamp"], sample_period["predicted_flow"], color="#2ca02c", linewidth=1.2, label="Predicted Valve Model (Cv)")
    ax_comp.plot(sample_period["Timestamp"], sample_period["sensor_uncalibrated"], color="#d62728", linestyle="--", linewidth=1.0, alpha=0.75, label="Physical Sensor (Assumed Inaccurate)")
    ax_comp.set_title("2-Week Profile Detail: Predicted Flow vs Physical Sensor", fontsize=11, fontweight="bold", pad=10)
    ax_comp.set_xlabel("Timestamp", fontsize=10, fontweight="bold")
    ax_comp.set_ylabel("Flow Rate", fontsize=10, fontweight="bold")
    ax_comp.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax_comp.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax_comp.xaxis.get_majorticklabels(), rotation=25, ha="right")
    ax_comp.legend(loc="upper left", frameon=True, facecolor="white", fontsize=9)
    ax_comp.grid(True, linestyle="--", alpha=0.5)

    ax_drift = fig.add_subplot(gs[2, 1])
    active_mask = (df["opening"] > 0.5) & (~df["drift_deviation"].isna())
    deviations = df.loc[active_mask, "drift_deviation"]

    ax_drift.hist(deviations, bins=60, color="#8e44ad", alpha=0.75, edgecolor="black", linewidth=0.5)
    mean_drift = deviations.mean()
    median_drift = deviations.median()
    ax_drift.axvline(mean_drift, color="red", linestyle="--", linewidth=1.5, label=f"Mean Deviation: +{mean_drift:.1f}")
    ax_drift.axvline(median_drift, color="orange", linestyle="-", linewidth=1.5, label=f"Median Deviation: +{median_drift:.1f}")
    ax_drift.set_title("Physical Sensor Deviation Distribution (Sensor Drift)", fontsize=11, fontweight="bold", pad=10)
    ax_drift.set_xlabel("Deviation (Physical Sensor - Valve Prediction)", fontsize=10, fontweight="bold")
    ax_drift.set_ylabel("Frequency", fontsize=10, fontweight="bold")
    ax_drift.legend(loc="upper right", frameon=True, facecolor="white", fontsize=9)
    ax_drift.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    plt.savefig(output_img, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Prediction overview plot saved to: '{output_img}'")
    return output_img


if __name__ == "__main__":
    est = ValveFlowEstimator(DEFAULT_CV_PATH)
    create_valve_characteristic_plot(est)
    create_prediction_overview_plot()

