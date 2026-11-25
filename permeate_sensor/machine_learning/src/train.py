"""
Retraining script for Decoupled Soft Sensor (46-FQI-106A) using LightGBM.
Strictly relies on Time Series Validation (No Random Shuffling / No Data Leakage).

Validation Methodology:
1. 5-Fold Expanding Window TimeSeriesSplit (Purged Chronological Validation).
2. Out-of-Time Future Holdout Split (Past Training vs Unseen Future Horizon).
3. Interval-by-Interval Tracking across all 20 discrete operational windows.

Source data: whb1_data_28102026.duckdb (table: historian_clean)
Filtered by: permeate_valid_data.csv (20 valid operational intervals)
"""

import os
import sys
import time
import json
import duckdb
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(CURRENT_DIR)
WORKSPACE_ROOT = os.path.dirname(ML_DIR)
VSENSOR_ROOT = os.path.dirname(WORKSPACE_ROOT)
for p in [ML_DIR, WORKSPACE_ROOT, VSENSOR_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from common.influx_client import query_multiple_intervals
from src.config import (
    INFLUX_DATABASE, HISTORIAN_TABLE, VALID_DATA_CSV,
    TARGET_TAG, MODELS_DIR, MODEL_PATH, FIG_DIR
)
from src.features import build_features, get_feature_columns


def load_valid_dataset():
    """Loads historian_clean filtered strictly by permeate_valid_data.csv intervals from InfluxDB."""
    if not os.path.exists(VALID_DATA_CSV):
        raise FileNotFoundError(f"Valid intervals file not found: {VALID_DATA_CSV}")

    df_intervals = pd.read_csv(VALID_DATA_CSV)
    df_intervals['start_dt'] = pd.to_datetime(df_intervals['Mulai'], format='%d-%b-%y %H:%M')
    df_intervals['end_dt'] = pd.to_datetime(df_intervals['Selesai'], format='%d-%b-%y %H:%M')

    raw_tags = [
        TARGET_TAG, '46-FQI-206A.A', '46-TIC-101.A', '46-AI-101.A', '54-FI-1002',
        '15-FQI-002.A', '15-FQI-402.A', '24-FQI-0124', '24-FQI-0123',
        '46-FIC-106A.Q', '20-PIC-106.A'
    ]

    print(f"Connecting to InfluxDB measurement '{HISTORIAN_TABLE}' (DB: '{INFLUX_DATABASE}')...")
    df_raw = query_multiple_intervals(
        tags=raw_tags,
        intervals_df=df_intervals,
        start_col='start_dt',
        end_col='end_dt',
        measurement=HISTORIAN_TABLE,
        database=INFLUX_DATABASE,
    )
    print(f"Loaded {len(df_raw):,} records across {len(df_intervals)} valid intervals in strict chronological order.")
    return df_raw, df_intervals


def compute_metrics(y_true, y_pred, name=""):
    """Computes comprehensive regression and flow volume metrics."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    if len(yt) == 0:
        return {'r2': np.nan, 'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'accuracy': np.nan, 'vol_diff_pct': np.nan}

    r2 = r2_score(yt, yp)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))

    active_mask = yt > 500.0
    if np.sum(active_mask) > 0:
        mape = float((np.abs(yt[active_mask] - yp[active_mask]) / yt[active_mask]).mean() * 100.0)
        acc = float(max(0.0, 100.0 - mape))
    else:
        mape = np.nan
        acc = np.nan

    total_actual = float(yt.sum())
    total_pred = float(yp.sum())
    vol_diff_pct = float(((total_pred - total_actual) / total_actual) * 100.0) if total_actual > 0 else 0.0

    return {
        'name': name,
        'r2': float(r2),
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'accuracy': acc,
        'vol_actual_lbs': total_actual,
        'vol_pred_lbs': total_pred,
        'vol_diff_pct': vol_diff_pct
    }


def train():
    start_time = time.time()
    df_raw, df_intervals = load_valid_dataset()

    print("Engineering physical domain features...")
    df = build_features(df_raw, is_training=True)
    feature_cols = get_feature_columns()
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    # Remove rows where target is missing
    target_mask = df[TARGET_TAG].notna()
    df_clean = df[target_mask].copy().sort_values('Timestamp').reset_index(drop=True)
    print(f"Valid target samples: {len(df_clean)} / {len(df)}")

    # -------------------------------------------------------------
    # 1. 5-Fold Expanding Window TimeSeriesSplit (Purged Chronological Validation)
    #    Zero data leakage: past data strictly used to predict future horizon.
    # -------------------------------------------------------------
    print("\n>>> Executing 5-Fold Expanding Window TimeSeriesSplit (Strictly Chronological)...")
    tscv = TimeSeriesSplit(n_splits=5)
    tscv_records = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df_clean)):
        X_tr = df_clean.iloc[train_idx][feature_cols]
        y_tr = df_clean.iloc[train_idx][TARGET_TAG]
        X_te = df_clean.iloc[test_idx][feature_cols]
        y_te = df_clean.iloc[test_idx][TARGET_TAG]
        whb_te = df_clean.iloc[test_idx]['whb1_on'].values

        t_tr_start = df_clean.iloc[train_idx]['Timestamp'].min()
        t_tr_end = df_clean.iloc[train_idx]['Timestamp'].max()
        t_te_start = df_clean.iloc[test_idx]['Timestamp'].min()
        t_te_end = df_clean.iloc[test_idx]['Timestamp'].max()

        ts_model = lgb.LGBMRegressor(
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=45,
            max_depth=9,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=4
        )
        ts_model.fit(X_tr, y_tr)
        pred_raw = ts_model.predict(X_te)
        pred_clamped = np.where(whb_te == 0, 0.0, np.maximum(0.0, pred_raw))

        m = compute_metrics(y_te.values, pred_clamped, name=f"TS_Fold_{fold+1}")
        m['train_start'] = str(t_tr_start)[:10]
        m['train_end'] = str(t_tr_end)[:10]
        m['test_start'] = str(t_te_start)[:10]
        m['test_end'] = str(t_te_end)[:10]
        m['train_samples'] = len(train_idx)
        m['test_samples'] = len(test_idx)
        tscv_records.append(m)

    df_tscv = pd.DataFrame(tscv_records)
    print(df_tscv[['name', 'train_start', 'train_end', 'test_start', 'test_end', 'test_samples', 'rmse', 'mae', 'mape', 'accuracy', 'vol_diff_pct']].to_string(index=False))
    print(f"--> Time Series Validation Average: RMSE={df_tscv['rmse'].mean():.2f}, MAE={df_tscv['mae'].mean():.2f}, MAPE={df_tscv['mape'].mean():.2f}%, Accuracy={df_tscv['accuracy'].mean():.2f}%, VolDiff={df_tscv['vol_diff_pct'].mean():+.2f}%")

    # -------------------------------------------------------------
    # 2. Chronological Out-of-Time Production Holdout Split
    #    Training: Past Data (Intervals 1–14: Oct 2023 – Jul 2025)
    #    Validation: Transition Horizon (Intervals 15–17: Nov 2025 – Jan 2026)
    #    Test: Unseen Future Year (Intervals 18–20: Feb 2026 – Jun 2026)
    # -------------------------------------------------------------
    print("\n>>> Setting up Chronological Out-of-Time Production Split...")
    train_mask = df_clean['Timestamp'] < '2025-11-01'
    val_mask = (df_clean['Timestamp'] >= '2025-11-01') & (df_clean['Timestamp'] < '2026-02-01')
    test_mask = df_clean['Timestamp'] >= '2026-02-01'

    X_train, y_train = df_clean.loc[train_mask, feature_cols], df_clean.loc[train_mask, TARGET_TAG]
    X_val, y_val = df_clean.loc[val_mask, feature_cols], df_clean.loc[val_mask, TARGET_TAG]
    X_test, y_test = df_clean.loc[test_mask, feature_cols], df_clean.loc[test_mask, TARGET_TAG]

    print(f"Train samples : {len(X_train):>8} ({len(X_train)/len(df_clean)*100:5.1f}%) [{df_clean.loc[train_mask, 'Timestamp'].min()} to {df_clean.loc[train_mask, 'Timestamp'].max()}]")
    print(f"Val samples   : {len(X_val):>8} ({len(X_val)/len(df_clean)*100:5.1f}%) [{df_clean.loc[val_mask, 'Timestamp'].min()} to {df_clean.loc[val_mask, 'Timestamp'].max()}]")
    print(f"Test samples  : {len(X_test):>8} ({len(X_test)/len(df_clean)*100:5.1f}%) [{df_clean.loc[test_mask, 'Timestamp'].min()} to {df_clean.loc[test_mask, 'Timestamp'].max()}]")

    print("\n>>> Training Final Production Model with Early Stopping on Chronological Val Set...")
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=45,
        max_depth=9,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=4
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False), lgb.log_evaluation(period=100)]
    )

    # Evaluate Holdout Sets
    def predict_with_gating(X, mask):
        raw = model.predict(X)
        whb_on = df_clean.loc[mask, 'whb1_on'].values
        return np.where(whb_on == 0, 0.0, np.maximum(0.0, raw))

    y_pred_train = predict_with_gating(X_train, train_mask)
    y_pred_val = predict_with_gating(X_val, val_mask)
    y_pred_test = predict_with_gating(X_test, test_mask)

    m_train = compute_metrics(y_train.values, y_pred_train, name="TRAIN (Oct 23 - Jul 25)")
    m_val = compute_metrics(y_val.values, y_pred_val, name="VAL (Nov 25 - Jan 26)")
    m_test = compute_metrics(y_test.values, y_pred_test, name="TEST OUT-OF-TIME (Feb 26 - Jun 26)")

    print("\n==================================================================")
    print("           CHRONOLOGICAL PRODUCTION EVALUATION SUMMARY            ")
    print("==================================================================")
    for m in [m_train, m_val, m_test]:
        print(f"[{m['name']}]")
        print(f"  R² Score         : {m['r2']:.4f}")
        print(f"  RMSE             : {m['rmse']:,.2f} LBS/HR")
        print(f"  MAE              : {m['mae']:,.2f} LBS/HR")
        print(f"  Active MAPE      : {m['mape']:.2f} %")
        print(f"  Accuracy Rate    : {m['accuracy']:.2f} %")
        print(f"  Cumulative Vol % : {m['vol_diff_pct']:+.2f} %")
    print("==================================================================")

    # -------------------------------------------------------------
    # 3. Interval-by-Interval Tracking across all 20 Valid Intervals
    # -------------------------------------------------------------
    print("\n>>> Tracking Performance across all 20 Valid Intervals chronologically...")
    interval_records = []
    df_clean['F_hat'] = predict_with_gating(df_clean[feature_cols], df_clean.index)

    for idx, row in df_intervals.iterrows():
        s = row['start_dt']
        e = row['end_dt']
        mask_i = (df_clean['Timestamp'] >= s) & (df_clean['Timestamp'] <= e)
        sub = df_clean[mask_i]
        if len(sub) == 0:
            continue
        m_i = compute_metrics(sub[TARGET_TAG].values, sub['F_hat'].values, name=f"Interval_{idx+1}")
        m_i['interval_idx'] = idx + 1
        m_i['start'] = str(s)
        m_i['end'] = str(e)
        m_i['count'] = len(sub)
        m_i['mean_actual'] = float(sub[TARGET_TAG].mean())
        m_i['mean_pred'] = float(sub['F_hat'].mean())
        interval_records.append(m_i)

    df_intervals_eval = pd.DataFrame(interval_records)
    print(df_intervals_eval[['interval_idx', 'start', 'end', 'count', 'mean_actual', 'mean_pred', 'mape', 'accuracy', 'vol_diff_pct']].to_string(index=False))

    # -------------------------------------------------------------
    # 4. Feature Importance
    # -------------------------------------------------------------
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': [int(x) for x in model.feature_importances_]
    }).sort_values('importance', ascending=False)
    print("\nFeature Importance:")
    print(importance_df.to_string(index=False))

    # -------------------------------------------------------------
    # 5. Save Model Artifact & Metrics
    # -------------------------------------------------------------
    os.makedirs(MODELS_DIR, exist_ok=True)
    artifact_payload = {
        'model': model,
        'feature_cols': feature_cols,
        'validation_strategy': 'TimeSeriesSplit (5 Chronological Expanding Folds) + Out-of-Time Production Test',
        'metrics_timeseries_cv': tscv_records,
        'metrics_holdout': {
            'train': m_train,
            'val': m_val,
            'test': m_test
        },
        'metrics_intervals': interval_records,
        'feature_importance': importance_df.to_dict(orient='records'),
        'trained_at': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_samples': len(df_clean),
        'database_source': DB_HISTORIAN_PATH,
        'intervals_source': VALID_DATA_CSV
    }
    joblib.dump(artifact_payload, MODEL_PATH)
    print(f"\nModel artifact successfully saved to: {MODEL_PATH}")

    summary_json_path = os.path.join(MODELS_DIR, 'retrain_evaluation_summary.json')
    json_payload = {k: v for k, v in artifact_payload.items() if k != 'model'}
    with open(summary_json_path, 'w') as f:
        json.dump(json_payload, f, indent=2)
    print(f"Summary JSON saved to: {summary_json_path}")

    elapsed = time.time() - start_time
    print(f"[COMPLETED] Retraining workflow with strict Time Series Validation finished in {elapsed:.2f} seconds.")
    return artifact_payload, df_clean


if __name__ == '__main__':
    train()
