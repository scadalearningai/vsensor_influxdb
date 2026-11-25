"""
Comprehensive Evaluation, Diagnostic Plotter & Report Generator
for Fuel Gas Soft Sensor (46-FIC-103A1.A) on explicit Train/Test period splits.
"""

import json
import os
import sys
from typing import Any, Dict

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(ML_DIR)
VSENSOR_ROOT = os.path.dirname(PROJECT_ROOT)
for p in (ML_DIR, PROJECT_ROOT, VSENSOR_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.timescale_client import query_dataframe as query_timescale_dataframe
from src.config import (
    DRIFT_REL_THRESHOLD,
    FIG_DIR,
    TIMESCALE_DATABASE,
    TIMESCALE_SCHEMA,
    MODEL_PATH,
    OUTPUT_DAILY_MEASUREMENT,
    OUTPUT_MEASUREMENT,
    REPORTS_DIR,
    SUMMARY_JSON_PATH,
    TARGET_TAG,
    TEST_DATA_CSV,
)
from src.features import build_features, get_feature_columns
from src.models import ML_CANDIDATE_NAMES
from src.train import apply_physics_gating, load_interval_dataset

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["axes.edgecolor"] = "#cbd5e1"
plt.rcParams["axes.linewidth"] = 0.8


def _plot_model_comparison(summary: Dict[str, Any], out_path: str) -> None:
    """Plots MAPE and RMSE comparison across candidate models (Train GroupKFold vs Blind Test splits)."""
    test_df = pd.DataFrame(summary["test_comparison"])
    ml_models = list(ML_CANDIDATE_NAMES)
    sub = test_df[test_df["model"].isin(ml_models)].copy()

    n_train_iv = len(summary["metrics_train_intervals"])
    gkf_rows = [
        {
            "model": m,
            "split": f"Train_{n_train_iv}Fold_GroupKFold",
            "mape": summary["group_cv_summary"][m]["mean_mape"],
            "rmse": summary["group_cv_summary"][m]["mean_rmse"],
        }
        for m in ml_models
    ]
    plot_df = pd.concat([pd.DataFrame(gkf_rows), sub[["model", "split", "mape", "rmse"]]], ignore_index=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.8), dpi=300)
    sns.barplot(data=plot_df, x="model", y="mape", hue="split", ax=ax1, palette=["#6366f1", "#0d9488", "#0284c7"])
    ax1.set_title("Model Comparison: Active MAPE (%) — Train CV vs Blind Test", fontsize=12, fontweight="bold")
    ax1.set_ylabel("MAPE (%) — Lower is Better", fontsize=10, fontweight="bold")
    ax1.set_xlabel("")
    ax1.tick_params(axis="x", rotation=12)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)

    sns.barplot(data=plot_df, x="model", y="rmse", hue="split", ax=ax2, palette=["#6366f1", "#0d9488", "#0284c7"])
    ax2.set_title("Model Comparison: RMSE (LBS/HR) — Train CV vs Blind Test", fontsize=12, fontweight="bold")
    ax2.set_ylabel("RMSE (LBS/HR) — Lower is Better", fontsize=10, fontweight="bold")
    ax2.set_xlabel("")
    ax2.tick_params(axis="x", rotation=12)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_interval_validation(summary: Dict[str, Any], out_path: str) -> None:
    """Plots mean flow and unseen accuracy across all Train (LOIO CV) and Blind Test intervals."""
    df_tr = pd.DataFrame(summary["metrics_train_intervals"]).copy()
    df_tr["group"] = "Train"
    df_tr["acc_plot"] = 100.0 - df_tr["unseen_cv_mape"]
    df_tr["label"] = [f"Tr{r['interval_idx']}\n({r['start'][:7]})" for _, r in df_tr.iterrows()]

    df_te = pd.DataFrame(summary["metrics_test_intervals"]).copy()
    df_te["group"] = "Test"
    df_te["acc_plot"] = df_te["accuracy"]
    df_te["label"] = [
        f"Te{r['interval_idx']}{'*' if r.get('is_last_period') else ''}\n({r['start'][:7]})"
        for _, r in df_te.iterrows()
    ]

    df_all = pd.concat([df_tr, df_te], ignore_index=True)
    x = np.arange(len(df_all))
    w = 0.36
    split_x = len(df_tr) - 0.5

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8.2), sharex=True, dpi=300, gridspec_kw={"height_ratios": [1.7, 1.3]})
    ax1.bar(x - w / 2, df_all["mean_actual"], w, label="Actual Sensor Mean (46-FIC-103A1.A)", color="#0f172a", alpha=0.9)
    ax1.bar(x + w / 2, df_all["mean_pred"], w, label=f"Soft Sensor Mean ({summary['model_name']})", color="#0284c7", alpha=0.9)
    ax1.axvline(split_x, color="#64748b", linestyle="--", lw=1.3)
    ax1.set_ylabel("Mean Fuel Gas Flow (LBS/HR)", fontsize=10.5, fontweight="bold")
    ax1.set_title(
        f"Interval Performance: {len(df_tr)} Train Intervals (LOIO CV) vs {len(df_te)} Blind Test Intervals (*Last Period)",
        fontsize=12.5,
        fontweight="bold",
        pad=10,
    )
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    bar_colors = np.where(df_all["acc_plot"] >= 90.0, "#10b981", "#f59e0b")
    bars = ax2.bar(x, df_all["acc_plot"], width=0.58, color=bar_colors, edgecolor="#0f172a", linewidth=0.5)
    ax2.axvline(split_x, color="#64748b", linestyle="--", lw=1.3, label="Train | Blind Test Boundary")
    ax2.axhline(90.0, color="#ef4444", linestyle="--", lw=1.2, label="90% Accuracy Target")
    ax2.set_ylabel("Unseen Accuracy (%)", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Operational Interval (Tr = Train LOIO CV, Te = Blind Test, Te* = Last Test Period)", fontsize=10.5, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_all["label"], fontsize=8.5)
    ax2.set_ylim(max(65, float(df_all["acc_plot"].min()) - 6), 103)
    ax2.legend(loc="lower left", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)
    for b in bars:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 0.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_holdout_parity_and_residuals(summary: Dict[str, Any], out_path: str) -> None:
    """Evaluates saved production model strictly on the blind Last Test Period of fuel_valid_test_data.csv."""
    df_raw, df_test_iv, _ = load_interval_dataset(TEST_DATA_CSV)
    df = build_features(df_raw, is_training=False)
    df = df[df[TARGET_TAG].notna()].copy().sort_values("Timestamp").reset_index(drop=True)
    last_idx = int(summary["last_test_period_info"]["interval_idx"])
    df_te = df[df["interval_idx"] == last_idx].copy().reset_index(drop=True)

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    winner_name = artifact["model_name"]

    pred_te = apply_physics_gating(model.predict(df_te[feature_cols]), df_te["fuel_gas_active"].to_numpy())
    actual_te = df_te[TARGET_TAG].to_numpy(dtype=float)
    residuals = actual_te - pred_te

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), dpi=300)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(actual_te), size=min(8000, len(actual_te)), replace=False)
    ax1.scatter(actual_te[idx], pred_te[idx], alpha=0.28, s=12, color="#0284c7", edgecolors="none", label="Blind Last Test Samples")
    lo = float(min(actual_te.min(), pred_te.min())) * 0.95
    hi = float(max(actual_te.max(), pred_te.max())) * 1.02
    ax1.plot([lo, hi], [lo, hi], color="#ef4444", linestyle="--", lw=1.6, label="1:1 Parity Line")
    ax1.set_xlabel("Actual Physical Flow 46-FIC-103A1.A (LBS/HR)", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Predicted Soft Sensor Flow (LBS/HR)", fontsize=10.5, fontweight="bold")
    ax1.set_title(
        f"Blind Last Test Parity ({winner_name}, {summary['last_test_period_info']['start'][:10]} .. {summary['last_test_period_info']['end'][:10]})",
        fontsize=11,
        fontweight="bold",
    )
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    sns.histplot(residuals, bins=60, kde=True, color="#0284c7", ax=ax2, edgecolor="#ffffff", lw=0.4)
    ax2.axvline(0.0, color="#ef4444", linestyle="--", lw=1.5, label="Zero Residual")
    ax2.axvline(float(np.mean(residuals)), color="#10b981", linestyle="-", lw=1.5, label=f"Mean Bias: {np.mean(residuals):+.1f} LBS/HR")
    ax2.set_xlabel("Residual (Actual - Predicted) [LBS/HR]", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Frequency", fontsize=10.5, fontweight="bold")
    ax2.set_title("Blind Last Test Residual Error Distribution", fontsize=11.5, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_feature_importance(summary: Dict[str, Any], out_path: str) -> None:
    """Plots horizontal bar chart of process tags by normalized importance."""
    imp_df = pd.DataFrame(summary["feature_importance"]).sort_values("importance_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    bars = ax.barh(imp_df["feature"], imp_df["importance_pct"], color="#0284c7", edgecolor="#0f172a", linewidth=0.5, height=0.6)
    for b in bars:
        w = b.get_width()
        ax.text(w + 0.5, b.get_y() + b.get_height() / 2, f"{w:.1f}%", va="center", ha="left", fontsize=9.5, fontweight="bold")
    ax.set_xlim(0, max(5.0, float(imp_df["importance_pct"].max())) * 1.18)
    ax.set_xlabel("Normalized Feature Importance (%)", fontsize=10.5, fontweight="bold")
    ax.set_title(f"Feature Importance ({len(imp_df)} Process Tags) — {summary['model_name']} Soft Sensor", fontsize=12.5, fontweight="bold", pad=12)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_historian_overview(summary: Dict[str, Any], out_path: str) -> None:
    """Plots full-historian inference overview and Blind Last Test window monitoring."""
    df_hr = query_timescale_dataframe(
        f'''
        SELECT time_bucket('1 hour', time) AS time,
               AVG(measured_fuelgas) AS measured,
               AVG(f_hat) AS f_hat,
               AVG(rel_dev_ewma) * 100.0 AS ewma_pct
        FROM "{TIMESCALE_SCHEMA}"."{OUTPUT_MEASUREMENT}"
        GROUP BY 1
        ORDER BY 1 ASC
        ''',
        database=TIMESCALE_DATABASE,
    )
    if not df_hr.empty:
        df_hr = df_hr.rename(columns={"Timestamp": "ts"})
    t_start = pd.to_datetime(summary["last_test_period_info"]["start"]).strftime("%Y-%m-%dT%H:%M:%SZ")
    t_end = pd.to_datetime(summary["last_test_period_info"]["end"]).strftime("%Y-%m-%dT%H:%M:%SZ")
    df_last = query_timescale_dataframe(
        f'''
        SELECT time, measured_fuelgas AS measured, f_hat, rel_dev_ewma * 100.0 AS ewma_pct
        FROM "{TIMESCALE_SCHEMA}"."{OUTPUT_MEASUREMENT}"
        WHERE time >= %s AND time <= %s
        ORDER BY time ASC
        ''',
        params=(t_start, t_end),
        database=TIMESCALE_DATABASE,
    )
    if not df_last.empty:
        df_last = df_last.rename(columns={"Timestamp": "ts"})

    fig = plt.figure(figsize=(15, 9.5), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.1])

    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(df_hr["ts"], df_hr["measured"], color="#0f172a", lw=0.7, alpha=0.75, label="Measured Physical Flow (46-FIC-103A1.A)")
    ax_top.plot(df_hr["ts"], df_hr["f_hat"], color="#0284c7", lw=0.8, alpha=0.85, label=f"ML Soft Sensor ({summary['model_name']})")
    ax_top.axvspan(pd.Timestamp(t_start), pd.Timestamp(t_end), color="#fef3c7", alpha=0.45, label="Last Test Period (Aug–Oct 2026)")
    ax_top.set_title("Full Historian Fuel Gas Flow (Hourly Mean, Oct 2022 – Nov 2026)", fontsize=12.5, fontweight="bold")
    ax_top.set_ylabel("Fuel Gas Flow (LBS/HR)", fontsize=10.5, fontweight="bold")
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_top.legend(loc="upper right", frameon=True)
    ax_top.grid(True, linestyle=":", alpha=0.6)

    ax_bl = fig.add_subplot(gs[1, 0])
    ax_bl.plot(df_last["ts"], df_last["measured"], color="#0f172a", lw=0.8, alpha=0.85, label="Physical 46-FIC-103A1.A")
    ax_bl.plot(df_last["ts"], df_last["f_hat"], color="#0284c7", lw=0.9, linestyle="--", alpha=0.9, label="ML Soft Sensor (F_hat)")
    ax_bl.axvline(pd.Timestamp("2026-09-09 08:00:00"), color="#ef4444", lw=1.1, linestyle=":", label="9 Sep 2026 08:00")
    ax_bl.set_title(f"Last Test Period: {t_start[:10]} – {t_end[:10]}", fontsize=11.5, fontweight="bold")
    ax_bl.set_ylabel("Flow (LBS/HR)", fontsize=10, fontweight="bold")
    ax_bl.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax_bl.legend(loc="lower right", fontsize=8.5, frameon=True)
    ax_bl.grid(True, linestyle=":", alpha=0.6)

    ax_br = fig.add_subplot(gs[1, 1])
    ax_br.plot(df_last["ts"], df_last["ewma_pct"], color="#8b5cf6", lw=1.0, label="EWMA Relative Deviation (%)")
    ax_br.axhline(DRIFT_REL_THRESHOLD * 100.0, color="#ef4444", lw=1.3, linestyle="--", label=f"Drift Alarm Threshold ({DRIFT_REL_THRESHOLD*100:.0f}%)")
    ax_br.set_title("EWMA Sensor Drift Monitor (Last Test Period)", fontsize=11.5, fontweight="bold")
    ax_br.set_ylabel("EWMA Deviation (%)", fontsize=10, fontweight="bold")
    ax_br.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax_br.legend(loc="upper right", fontsize=8.5, frameon=True)
    ax_br.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def _write_markdown_report(summary: Dict[str, Any], sep9_stats: Dict[str, Any]) -> str:
    """Writes concise engineering evaluation markdown report based strictly on computed metrics."""
    wm = summary["winner_metrics"]
    last_t = wm["test_last_period"]
    all_t = wm["test_all_periods"]
    gkf = wm["train_group_cv_mean"]
    ci = wm["test_last_95ci"]
    last_info = summary["last_test_period_info"]

    lb_rows = "\n".join(
        f"| {r['model']} | {r['train_groupcv_mape']:.2f}% ({r['train_groupcv_rmse']:.1f}) | "
        f"{r['test_last_mape']:.2f}% ({r['test_last_rmse']:.1f}, R²={r['test_last_r2']:.3f}) | "
        f"{r['test_all_mape']:.2f}% ({r['test_all_rmse']:.1f}) |"
        for r in summary["model_leaderboard"]
    )
    train_int_rows = "\n".join(
        f"| Train Int {r['interval_idx']} | {r['start']} | {r['end']} | {r['count']:,} | {r['mean_actual']:.1f} | {r['mean_pred']:.1f} | {r['unseen_cv_rmse']:.2f} | {r['unseen_cv_mape']:.2f}% |"
        for r in summary["metrics_train_intervals"]
    )
    test_int_rows = "\n".join(
        f"| Test Int {r['interval_idx']}{' **(Last)**' if r.get('is_last_period') else ''} | {r['start']} | {r['end']} | {r['count']:,} | {r['mean_actual']:.1f} | {r['mean_pred']:.1f} | {r['rmse']:.2f} | {r['mape']:.2f}% | {r['vol_diff_pct']:+.2f}% |"
        for r in summary["metrics_test_intervals"]
    )
    slice_rows = "\n".join(
        f"| {r['slice']} | {r['count']:,} | {r['rmse']:.2f} | {r['mae']:.2f} | {r['mape']:.2f}% | {r['vol_diff_pct']:+.2f}% |"
        for r in summary["slice_errors_last_test"]
    )
    top_feats = "\n".join(
        f"| {i+1} | `{r['feature']}` | {r['importance_pct']:.2f}% |"
        for i, r in enumerate(summary["feature_importance"])
    )
    meta_weights = summary.get("meta_weights")
    meta_section = ""
    if meta_weights:
        mw_rows = "\n".join(f"| `{k}` | {v:+.4f} |" for k, v in meta_weights.items())
        meta_section = f"\n### GroupKFold Meta-Learner Weights (`StackedEnsemble`)\n\n| Base Estimator | RidgeCV Weight |\n| :--- | :---: |\n{mw_rows}\n"

    n_feats = len(summary["feature_cols"])
    n_train_iv = len(summary["metrics_train_intervals"])
    n_test_iv = len(summary["metrics_test_intervals"])
    report = f"""# Fuel Gas Soft Sensor (`46-FIC-103A1.A`) — Retraining & Blind Test Report

- **Trained At**: {summary['trained_at']}
- **Database Source**: `whb1_data2023-2026.duckdb` (`5-minute resolution`)
- **Selected Model**: `{summary['model_name']}` ({n_feats} Raw Process Features; selected strictly via {n_train_iv}-Fold Train `GroupKFold`)
- **Feature Tags**: {", ".join(f"`{t}`" for t in summary['feature_cols'])}
- **Training Dataset (`fuel_valid_train_data.csv`)**: {summary['total_train_samples']:,} valid rows across {n_train_iv} intervals ({summary['dropped_null_train_rows']} null-target rows removed)
- **Blind Test Dataset (`fuel_valid_test_data.csv`)**: {summary['total_test_samples']:,} valid rows across {n_test_iv} unique intervals ({summary['deduplicated_test_intervals']} duplicate interval deduplicated, {summary['dropped_null_test_rows']} null-target rows removed)
- **Primary Test (Last Period)**: Test Int {last_info['interval_idx']} (`{last_info['start']}` .. `{last_info['end']}`, `n = {last_info['count']:,}`)
- **Model Artifact**: `{MODEL_PATH}`
- **Batch Output Database**: `{DB_OUTPUT_PATH}`

## 1. Multi-Model Benchmark Leaderboard (Zero-Leakage Train Selection vs Blind Test)

| Model | {n_train_iv}-Fold Train `GroupKFold` MAPE (RMSE) | Blind Last Test Period MAPE (RMSE, R²) | Blind All Test Periods MAPE (RMSE) |
| :--- | :---: | :---: | :---: |
{lb_rows}

## 2. Selected Winner Performance (`{summary['model_name']}`)

- **Primary Blind Holdout — Last Test Period (`{last_info['start']}` .. `{last_info['end']}`)**:
  - R²: **{last_t['r2']:.4f}**
  - RMSE: **{last_t['rmse']:.2f} LBS/HR** (95% Block-Bootstrap CI: `[{ci['rmse_95ci'][0]:.2f}, {ci['rmse_95ci'][1]:.2f}]`)
  - MAE: **{last_t['mae']:.2f} LBS/HR** (95% CI: `[{ci['mae_95ci'][0]:.2f}, {ci['mae_95ci'][1]:.2f}]`)
  - Active MAPE: **{last_t['mape']:.2f}%** (Accuracy: **{last_t['accuracy']:.2f}%**, 95% CI: `[{ci['mape_95ci'][0]:.2f}%, {ci['mape_95ci'][1]:.2f}%]`)
  - Cumulative Volume Deviation: **{last_t['vol_diff_pct']:+.2f}%**
- **All {n_test_iv} Blind Test Periods Combined (`fuel_valid_test_data.csv`)**:
  - R²: **{all_t['r2']:.4f}**, RMSE: **{all_t['rmse']:.2f} LBS/HR**, MAE: **{all_t['mae']:.2f} LBS/HR**, MAPE: **{all_t['mape']:.2f}%**, Vol Diff: **{all_t['vol_diff_pct']:+.2f}%**
- **{n_train_iv}-Fold Leave-One-Interval-Out (`GroupKFold`) on Train Data**:
  - RMSE: **{gkf['mean_rmse']:.2f} LBS/HR**, MAE: **{gkf['mean_mae']:.2f} LBS/HR**, MAPE: **{gkf['mean_mape']:.2f}%**, Accuracy: **{gkf['mean_accuracy']:.2f}%**

### Valve-Opening Slice Error Analysis (Last Test Period)

| Valve Opening Slice | Count | RMSE (LBS/HR) | MAE (LBS/HR) | Active MAPE | Vol Diff |
| :---: | :---: | :---: | :---: | :---: | :---: |
{slice_rows}

## 3. Per-Interval Performance Breakdown

### 3A. Training Intervals (`fuel_valid_train_data.csv` — Unseen LOIO CV)

| Interval | Start | End | Rows | Mean Actual | Mean Refit Pred | Unseen CV RMSE | Unseen CV MAPE |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
{train_int_rows}

### 3B. Blind Test Intervals (`fuel_valid_test_data.csv` — 100% Out-of-Sample)

| Interval | Start | End | Rows | Mean Actual | Mean Pred | Blind RMSE | Blind MAPE | Vol Diff |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{test_int_rows}

## 4. Feature Importance ({n_feats} Process Tags)

| Rank | Feature | Normalized Importance |
| :---: | :--- | :---: |
{top_feats}
{meta_section}
## 5. Full Historian Inference & 9 September 2026 Snapshot

- **Total Historian Rows Evaluated**: {sep9_stats['total_rows']:,} (`Oct 2022 – Nov 2026`, 5-minute grid)
- **9 September 2026 Daily Summary** (`dt = 5 min` mass integration):
  - Measured Total Mass: **{sep9_stats['measured_total_lbs']:,.2f} LBS**
  - Soft Sensor Total Mass: **{sep9_stats['softsensor_total_lbs']:,.2f} LBS**
  - Daily Difference: **{sep9_stats['diff_pct']:+.2f}%**
"""
    report_path = os.path.join(REPORTS_DIR, "TRAINING_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report_path


def run_evaluation() -> None:
    print("=" * 75)
    print("FUEL GAS SOFT SENSOR - EVALUATION FIGURES & REPORT GENERATION")
    print("=" * 75)
    os.makedirs(FIG_DIR, exist_ok=True)
    with open(SUMMARY_JSON_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)

    p1 = os.path.join(FIG_DIR, "model_benchmark_comparison.png")
    _plot_model_comparison(summary, p1)
    print(f"Saved: {p1}")

    p2 = os.path.join(FIG_DIR, "interval_performance_validation.png")
    _plot_interval_validation(summary, p2)
    print(f"Saved: {p2}")

    p3 = os.path.join(FIG_DIR, "parity_residuals_holdout.png")
    _plot_holdout_parity_and_residuals(summary, p3)
    print(f"Saved: {p3}")

    p4 = os.path.join(FIG_DIR, "feature_importance.png")
    _plot_feature_importance(summary, p4)
    print(f"Saved: {p4}")

    p5 = os.path.join(FIG_DIR, "historian_drift_overview.png")
    _plot_historian_overview(summary, p5)
    print(f"Saved: {p5}")

    df_count = query_timescale_dataframe(
        f'SELECT COUNT(f_hat) AS total_rows FROM "{TIMESCALE_SCHEMA}"."{OUTPUT_MEASUREMENT}"',
        database=TIMESCALE_DATABASE,
    )
    total_rows = int(df_count["total_rows"].iloc[0]) if not df_count.empty else 0
    df_sep9 = query_timescale_dataframe(
        f'''
        SELECT * FROM "{TIMESCALE_SCHEMA}"."{OUTPUT_DAILY_MEASUREMENT}"
        WHERE time >= '2026-09-09T00:00:00Z' AND time < '2026-09-10T00:00:00Z'
        LIMIT 1
        ''',
        database=TIMESCALE_DATABASE,
    )
    if not df_sep9.empty:
        sep9 = df_sep9.iloc[0].to_dict()
        sep9_stats = {
            "total_rows": total_rows,
            "measured_total_lbs": float(sep9["measured_total_lbs"]),
            "softsensor_total_lbs": float(sep9["softsensor_total_lbs"]),
            "diff_pct": float(sep9["diff_pct"]),
        }
    else:
        sep9_stats = {
            "total_rows": total_rows,
            "measured_total_lbs": 0.0,
            "softsensor_total_lbs": 0.0,
            "diff_pct": 0.0,
        }

    report_path = _write_markdown_report(summary, sep9_stats)
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    run_evaluation()
