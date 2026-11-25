"""
Comprehensive Evaluation & Visualization Generator for Decoupled Soft Sensor.
Uses the EXACT SAME dataset:
- Training Data   : Agustus 2025 - 31 Mei 2026 (437.199 baris)
- Out-of-Time Test: 1 Juni 2026 - 20 September 2026 (149.873 baris)
- Operational Validation Focus: September 2026 (5 - 12 September 2026 & Daily Report 9 September 2026)
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(CURRENT_DIR)
WORKSPACE_ROOT = os.path.dirname(ML_DIR)
VSENSOR_ROOT = os.path.dirname(WORKSPACE_ROOT)
for p in [ML_DIR, WORKSPACE_ROOT, VSENSOR_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from common.influx_client import query_dataframe

from src.config import (
    INFLUX_DATABASE, HISTORIAN_TABLE, TARGET_TAG,
    FIG_DIR, MODELS_DIR, MODEL_PATH
)
from src.features import build_features, get_feature_columns
from src.inference import SoftSensorEngine

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8

def run_evaluation():
    print(f"Loading test data from InfluxDB measurement '{HISTORIAN_TABLE}' (DB: '{INFLUX_DATABASE}')...")
    df_raw = query_dataframe(f'SELECT * FROM "{HISTORIAN_TABLE}" WHERE time >= \'2026-06-01T00:00:00Z\'', database=INFLUX_DATABASE)
    if df_raw.empty:
        raise RuntimeError(f"No evaluation data found in InfluxDB '{INFLUX_DATABASE}.{HISTORIAN_TABLE}'.")
    print(f"Loaded {len(df_raw):,} records from InfluxDB.")

    print("Executing feature extraction for decoupled model (10 features)...")
    df10 = build_features(df_raw, is_training=False)

    m10_file = MODEL_PATH
    engine10 = SoftSensorEngine(m10_file)

    cols10 = get_feature_columns()

    # Predictions
    pred10_df = engine10.predict_dataframe(df_raw)
    df_raw['F_hat'] = pred10_df['F_hat']
    df_raw['whb1_on'] = pred10_df['whb1_on']
    df_raw['ewma_drift'] = pred10_df.get('ewma_drift', 0.0)

    ts = pd.to_datetime(df_raw['Timestamp'])
    y_test = df_raw[TARGET_TAG]
    F_hat = df_raw['F_hat']

    # -------------------------------------------------------------
    # Figure 1: Time Series Comparison (September 2026 matching Daily Report)
    # Jendela 5 - 12 September 2026
    # -------------------------------------------------------------
    print("Generating Figure 1: Time Series Comparison (5 - 12 September 2026)...")
    sample_mask = (ts >= '2026-09-05 00:00:00') & (ts <= '2026-09-12 23:59:59')
    df_sub = df_raw[sample_mask]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
    
    # Subplot 1: Flow comparison
    ax1.plot(pd.to_datetime(df_sub['Timestamp']), df_sub[TARGET_TAG], label='Actual Physical Sensor (46-FQI-106A)', color='#0f172a', lw=1.3, alpha=0.85)
    ax1.plot(pd.to_datetime(df_sub['Timestamp']), df_sub['F_hat'], label='Decoupled Soft Sensor (10 Features)', color='#0284c7', lw=1.6, linestyle='--')
    
    ax1.axvline(pd.Timestamp('2026-09-09 08:00:00'), color='#ef4444', linestyle='-', lw=1.2, alpha=0.75, label='Medco Daily Production Report Point (9 Sep 08:00 AM)')
    ax1.set_ylabel('Permeate Flow Rate (LBS/HR)', fontsize=11, fontweight='bold', color='#1e293b')
    ax1.set_title('Operational Validation: Soft Sensor vs Physical Sensor (5 – 12 September 2026 Window)', fontsize=13, fontweight='bold', color='#0f172a', pad=12)
    ax1.legend(loc='upper right', frameon=True, facecolor='#ffffff', framealpha=0.9, fontsize=9.5)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Subplot 2: Residual comparison
    res = df_sub[TARGET_TAG] - df_sub['F_hat']
    ax2.plot(pd.to_datetime(df_sub['Timestamp']), res, color='#0284c7', lw=1.1, label='Residual (Actual − Soft Sensor Prediction)')
    ax2.fill_between(pd.to_datetime(df_sub['Timestamp']), 0, res, color='#0284c7', alpha=0.15)
    ax2.axhline(0, color='#64748b', linestyle='--', lw=1)
    ax2.set_ylabel('Residual (LBS/HR)', fontsize=10, fontweight='bold', color='#1e293b')
    ax2.set_xlabel('Operational Timestamp (September 2026)', fontsize=11, fontweight='bold', color='#1e293b')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %H:%M'))
    ax2.legend(loc='lower right', frameon=True, fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    fig1_path = os.path.join(FIG_DIR, 'evaluasi_model_timeseries.png')
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig1_path}")

    # -------------------------------------------------------------
    # Figure 2: Feature Importance Bar Chart
    # -------------------------------------------------------------
    print("Generating Figure 2: Feature Importance...")
    artifact = joblib.load(m10_file)
    imp_df = pd.DataFrame(artifact['feature_importance']).sort_values('importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#38bdf8' if 'temp' in f or 'o2' in f or '54-' in f or '46-' in f else '#0284c7' for f in imp_df['feature']]
    bars = ax.barh(imp_df['feature'], imp_df['importance'], color=colors, height=0.65, edgecolor='#0f172a', lw=0.5)
    ax.set_xlabel('Feature Importance (Split Count)', fontsize=11, fontweight='bold', color='#1e293b')
    ax.set_ylabel('Model Features', fontsize=11, fontweight='bold', color='#1e293b')
    ax.set_title('Feature Importance Distribution — Permeate Gas Soft Sensor (LightGBM)', fontsize=13, fontweight='bold', color='#0f172a', pad=12)
    ax.grid(True, axis='x', linestyle=':', alpha=0.6)
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 5, bar.get_y() + bar.get_height()/2, f'{int(w)}', va='center', ha='left', fontsize=9, color='#1e293b', fontweight='bold')

    ax.set_xlim(0, imp_df['importance'].max() * 1.15)
    plt.tight_layout()
    fig2_path = os.path.join(FIG_DIR, 'feature_importance.png')
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path}")

    # -------------------------------------------------------------
    # Figure 3: Parity Scatter Plot & Residual Distribution (Full Out-of-Time Test Set)
    # -------------------------------------------------------------
    print("Generating Figure 3: Parity Plot & Residual Distribution (Full Test Set 1 Juni - 20 September 2026)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sample_idx = np.random.choice(len(df_raw), size=min(6000, len(df_raw)), replace=False)
    ax1.scatter(y_test.iloc[sample_idx], F_hat.iloc[sample_idx], alpha=0.25, color='#0284c7', s=12, edgecolors='none', label='Prediction Samples')
    max_val = max(y_test.max(), F_hat.max())
    ax1.plot([0, max_val], [0, max_val], color='#ef4444', linestyle='--', lw=1.8, label='1:1 Ideal Parity Line')
    ax1.set_xlabel('Actual Physical Flow Rate (LBS/HR)', fontsize=11, fontweight='bold', color='#1e293b')
    ax1.set_ylabel('Soft Sensor Predicted Flow Rate (LBS/HR)', fontsize=11, fontweight='bold', color='#1e293b')
    ax1.set_title('Parity Plot (Actual vs Predicted Flow Rate) — Test Set', fontsize=12, fontweight='bold', color='#0f172a', pad=10)
    ax1.legend(loc='upper left', frameon=True, fontsize=9.5)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Residual distribution
    active_mask = y_test > 500.0
    residuals = y_test[active_mask] - F_hat[active_mask]
    sns.histplot(residuals, bins=60, kde=True, color='#0284c7', ax=ax2, edgecolor='#ffffff', lw=0.5)
    ax2.axvline(0, color='#ef4444', linestyle='--', lw=1.5, label='Zero Error (0)')
    ax2.axvline(residuals.mean(), color='#10b981', linestyle='-', lw=1.5, label=f'Mean Bias ({residuals.mean():.1f} LBS/HR)')
    ax2.set_xlabel('Residual Error (LBS/HR)', fontsize=11, fontweight='bold', color='#1e293b')
    ax2.set_ylabel('Data Frequency', fontsize=11, fontweight='bold', color='#1e293b')
    ax2.set_title('Residual Error Distribution (Active Burner Condition)', fontsize=12, fontweight='bold', color='#0f172a', pad=10)
    ax2.legend(loc='upper right', frameon=True, fontsize=9.5)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    fig3_path = os.path.join(FIG_DIR, 'parity_scatter_residual.png')
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved: {fig3_path}")

    # -------------------------------------------------------------
    # Figure 4: Drift Detection Demonstration (September 2026 Window)
    # -------------------------------------------------------------
    print("Generating Figure 4: Sensor Drift Demonstration (September 2026 Window)...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # 5 - 15 September 2026
    drift_period = (ts >= '2026-09-05 00:00:00') & (ts <= '2026-09-15 23:59:59')
    df_d = df_raw[drift_period]

    ax1.plot(pd.to_datetime(df_d['Timestamp']), df_d[TARGET_TAG], label='Fisik 46-FQI-106A', color='#0f172a', lw=1.2)
    ax1.plot(pd.to_datetime(df_d['Timestamp']), df_d['F_hat'], label='Soft Sensor (F_hat)', color='#0284c7', lw=1.5, linestyle='--')
    ax1.axvline(pd.Timestamp('2026-09-09 08:00:00'), color='#ef4444', linestyle=':', lw=1.2, label='Titik Daily Report (9 Sep)')
    ax1.set_ylabel('Laju Alir (LBS/HR)', fontsize=10, fontweight='bold')
    ax1.set_title('Monitoring Kesehatan Sensor & Indikator Drift (September 2026)', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.plot(pd.to_datetime(df_d['Timestamp']), df_d['ewma_drift'] * 100.0, color='#8b5cf6', lw=1.5, label='Deviasi Relatif EWMA (%)')
    ax2.axhline(15.0, color='#ef4444', linestyle='--', lw=1.5, label='Ambang Alarm Drift (15%)')
    ax2.set_ylabel('EWMA Deviasi (%)', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Waktu (September 2026)', fontsize=10, fontweight='bold')
    ax2.set_ylim(0, 30)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    fig4_path = os.path.join(FIG_DIR, 'drift_detection_demo.png')
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Saved: {fig4_path}")

    print("All figures successfully regenerated matching exact validation period.")

if __name__ == '__main__':
    run_evaluation()
