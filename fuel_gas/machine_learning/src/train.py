"""
Training, Benchmarking & Blind Holdout Evaluation Pipeline for Fuel Gas Soft Sensor (46-FIC-103A1.A).

Validation & Testing Methodology (Zero Leakage):
1. Load explicit training intervals from `fuel_valid_train_data.csv` (deduplicated).
2. Benchmark all 4 candidates strictly within `fuel_valid_train_data.csv` via:
   - Leave-One-Interval-Out (`GroupKFold` across all training intervals)
   Select winning model purely on unseen GroupKFold CV MAPE/RMSE (zero test peeking).
3. Fit each candidate on 100% of `fuel_valid_train_data.csv` and evaluate blind on `fuel_valid_test_data.csv`:
   - `Test_LastPeriod`: Last chronological interval in `fuel_valid_test_data.csv` (primary blind holdout)
   - `Test_AllPeriods`: All unique intervals in `fuel_valid_test_data.csv` + per-interval breakdown
4. Run 5-fold expanding `TimeSeriesSplit`, 95% Block-Bootstrap CI, and valve-opening slice error analysis
   on `Test_LastPeriod`, then export the final model trained strictly on `fuel_valid_train_data.csv`.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, TimeSeriesSplit

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(ML_DIR)
VSENSOR_ROOT = os.path.dirname(PROJECT_ROOT)
for p in (ML_DIR, PROJECT_ROOT, VSENSOR_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.influx_client import query_multiple_intervals
from src.config import (
    INFLUX_DATABASE,
    HISTORIAN_TABLE,
    MIN_ACTIVE_FLOW,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    REQUIRED_RAW_TAGS,
    SUMMARY_JSON_PATH,
    TARGET_TAG,
    TEST_DATA_CSV,
    TRAIN_DATA_CSV,
)
from src.features import build_features, get_feature_columns
from src.models import GroupStackedRegressor, ML_CANDIDATE_NAMES, build_candidate_models


def load_interval_dataset(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Loads historian_clean filtered strictly by deduplicated intervals in csv_path from InfluxDB.
    Returns (df_raw, df_intervals_unique, num_duplicate_intervals_removed).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Intervals file not found: {csv_path}")

    df_intervals = pd.read_csv(csv_path)
    df_intervals["Mulai"] = df_intervals["Mulai"].astype(str).str.strip()
    df_intervals["Selesai"] = df_intervals["Selesai"].astype(str).str.strip()
    df_intervals["start_dt"] = pd.to_datetime(df_intervals["Mulai"], format="%d-%b-%y %H:%M")
    df_intervals["end_dt"] = pd.to_datetime(df_intervals["Selesai"], format="%d-%b-%y %H:%M")

    raw_len = len(df_intervals)
    df_intervals = (
        df_intervals.sort_values(["start_dt", "end_dt"])
        .drop_duplicates(subset=["start_dt", "end_dt"])
        .reset_index(drop=True)
    )
    dup_removed = raw_len - len(df_intervals)

    select_cols = [TARGET_TAG] + REQUIRED_RAW_TAGS
    df_raw = query_multiple_intervals(
        tags=select_cols,
        intervals_df=df_intervals,
        start_col="start_dt",
        end_col="end_dt",
        measurement=HISTORIAN_TABLE,
        database=INFLUX_DATABASE,
    )

    df_raw["interval_idx"] = 0
    for idx, row in df_intervals.iterrows():
        mask = (df_raw["Timestamp"] >= row.start_dt) & (df_raw["Timestamp"] <= row.end_dt)
        df_raw.loc[mask, "interval_idx"] = idx + 1

    return df_raw, df_intervals, dup_removed


def load_valid_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Backward-compatible helper loading deduplicated training intervals."""
    df_raw, df_intervals, _ = load_interval_dataset(TRAIN_DATA_CSV)
    return df_raw, df_intervals


def apply_physics_gating(pred_raw: np.ndarray, active_flag: np.ndarray) -> np.ndarray:
    """Clamps prediction to 0.0 when burner or fuel valve is closed, and floors negatives."""
    return np.where(active_flag == 0, 0.0, np.maximum(0.0, pred_raw))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str = "") -> Dict[str, Any]:
    """Computes regression, active-region MAPE, and cumulative volume deviation metrics."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp)) if len(yt) > 1 and np.var(yt) > 1e-12 else 0.0

    active = yt >= MIN_ACTIVE_FLOW
    if np.any(active):
        mape = float(np.mean(np.abs(yt[active] - yp[active]) / yt[active]) * 100.0)
        acc = float(100.0 - mape)
    else:
        mape = 0.0
        acc = 100.0

    total_actual = float(np.sum(yt))
    total_pred = float(np.sum(yp))
    vol_diff_pct = float(((total_pred - total_actual) / max(total_actual, 1.0)) * 100.0)

    return {
        "name": name,
        "count": int(len(yt)),
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "accuracy": acc,
        "vol_actual_lbs": total_actual,
        "vol_pred_lbs": total_pred,
        "vol_diff_pct": vol_diff_pct,
    }


def block_bootstrap_ci(
    y_true: np.ndarray, y_pred: np.ndarray, block_size: int = 12, n_boot: int = 400, seed: int = 42
) -> Dict[str, List[float]]:
    """Computes 95% confidence intervals via hourly block bootstrap (12 x 5-min = 1h)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    n_blocks = max(1, n // block_size)
    blocks = [(i * block_size, min((i + 1) * block_size, n)) for i in range(n_blocks)]

    rmses, maes, mapes = [], [], []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        idx = np.concatenate([np.arange(blocks[b][0], blocks[b][1]) for b in chosen])
        yt, yp = y_true[idx], y_pred[idx]
        rmses.append(float(np.sqrt(mean_squared_error(yt, yp))))
        maes.append(float(mean_absolute_error(yt, yp)))
        active = yt >= MIN_ACTIVE_FLOW
        if np.any(active):
            mapes.append(float(np.mean(np.abs(yt[active] - yp[active]) / yt[active]) * 100.0))

    return {
        "rmse_95ci": [float(np.percentile(rmses, 2.5)), float(np.percentile(rmses, 97.5))],
        "mae_95ci": [float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))],
        "mape_95ci": [float(np.percentile(mapes, 2.5)), float(np.percentile(mapes, 97.5))] if mapes else [None, None],
    }


def fit_estimator(estimator: Any, df_train: pd.DataFrame, feature_cols: List[str]) -> Any:
    """Fits an estimator, passing operational interval groups to GroupStackedRegressor."""
    X_tr = df_train[feature_cols]
    y_tr = df_train[TARGET_TAG].to_numpy(dtype=float)
    if isinstance(estimator, GroupStackedRegressor):
        groups = df_train["interval_idx"].to_numpy() if "interval_idx" in df_train.columns else None
        estimator.fit(X_tr, y_tr, groups=groups)
    else:
        estimator.fit(X_tr, y_tr)
    return estimator


def eval_estimator(
    estimator: Any,
    model_name: str,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    split_label: str,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Evaluates an already fitted estimator on df_test with deterministic physics gating."""
    raw_pred = estimator.predict(df_test[feature_cols])
    pred = apply_physics_gating(raw_pred, df_test["fuel_gas_active"].to_numpy())
    metrics = compute_metrics(df_test[TARGET_TAG].to_numpy(dtype=float), pred, name=f"{model_name} | {split_label}")
    metrics["model"] = model_name
    metrics["split"] = split_label
    return metrics, pred


def fit_and_eval_split(
    model_name: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    split_label: str,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Fits a fresh instance of model_name strictly on df_train and evaluates on df_test."""
    estimator = fit_estimator(build_candidate_models()[model_name], df_train, feature_cols)
    return eval_estimator(estimator, model_name, df_test, feature_cols, split_label)


def _unwrap_feature_importances(estimator: Any) -> Optional[np.ndarray]:
    """Unwraps _ArrayPipeline / sklearn Pipeline to extract native tree feature importances if present."""
    inner = getattr(estimator, "estimator", estimator)
    if hasattr(inner, "named_steps") and "model" in inner.named_steps:
        inner = inner.named_steps["model"]
    if hasattr(inner, "feature_importances_"):
        return np.asarray(inner.feature_importances_, dtype=float)
    return None


def extract_feature_importance(
    model: Any,
    feature_cols: List[str],
    df_eval: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Extracts normalized feature importance via native tree splits or model-agnostic permutation importance."""
    native_imp = _unwrap_feature_importances(model)
    if native_imp is not None and native_imp.sum() > 0:
        scores = native_imp / native_imp.sum()
    else:
        sample_n = min(4000, len(df_eval))
        idx = np.linspace(0, len(df_eval) - 1, sample_n, dtype=int)
        X_sub = df_eval.iloc[idx][feature_cols].to_numpy(dtype=float)
        y_sub = df_eval.iloc[idx][TARGET_TAG].to_numpy(dtype=float)
        perm = permutation_importance(model, X_sub, y_sub, n_repeats=5, random_state=RANDOM_SEED, n_jobs=1)
        raw_scores = np.maximum(0.0, perm.importances_mean)
        scores = raw_scores / max(float(raw_scores.sum()), 1e-9)

    df_imp = pd.DataFrame({"feature": feature_cols, "importance_pct": (scores * 100.0).round(2)})
    return df_imp.sort_values("importance_pct", ascending=False).to_dict(orient="records")


def compute_slice_analysis(df_test: pd.DataFrame, y_pred: np.ndarray) -> List[Dict[str, Any]]:
    """Evaluates error across valve-opening operating regimes on blind holdout test data."""
    sub = df_test[["46-FIC-103A1.Q", TARGET_TAG]].copy()
    sub["pred"] = y_pred
    sub["q_bin"] = pd.cut(sub["46-FIC-103A1.Q"], bins=[0, 20, 22, 24, 100], labels=["<=20%", "20-22%", "22-24%", ">24%"])
    slices = []
    for label, grp in sub.groupby("q_bin", observed=False):
        if len(grp) == 0:
            continue
        m = compute_metrics(grp[TARGET_TAG].to_numpy(), grp["pred"].to_numpy(), name=f"Valve {label}")
        m["slice"] = str(label)
        slices.append(m)
    return slices


def _prepare_clean_dataset(csv_path: str, label: str) -> Tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """Loads, builds features, drops null targets, and logs dataset health."""
    df_raw, df_intervals, dup_cnt = load_interval_dataset(csv_path)
    null_target_cnt = int(df_raw[TARGET_TAG].isna().sum())
    df_feat = build_features(df_raw, is_training=(label == "TRAIN"))
    df_clean = df_feat[df_feat[TARGET_TAG].notna()].copy().sort_values("Timestamp").reset_index(drop=True)
    dup_note = f" (deduplicated {dup_cnt} duplicate interval row)" if dup_cnt > 0 else ""
    print(
        f"[{label}] Loaded {len(df_raw):,} raw rows across {len(df_intervals)} intervals{dup_note} "
        f"-> {len(df_clean):,} valid samples ({null_target_cnt} null targets dropped)."
    )
    return df_clean, df_intervals, null_target_cnt, dup_cnt


def train() -> Tuple[Dict[str, Any], pd.DataFrame]:
    t0 = time.time()
    print("=" * 75)
    print("FUEL GAS SOFT SENSOR (46-FIC-103A1.A) - ML TRAINING & BLIND TESTING")
    print("=" * 75)

    feature_cols = get_feature_columns("selected")
    df_train, df_train_iv, null_train_cnt, _ = _prepare_clean_dataset(TRAIN_DATA_CSV, "TRAIN")
    df_test, df_test_iv, null_test_cnt, test_dup_cnt = _prepare_clean_dataset(TEST_DATA_CSV, "TEST")

    feat_null_pct = (df_train[feature_cols].isna().mean() * 100).round(4)
    nonzero_nulls = feat_null_pct[feat_null_pct > 0]
    print("Train feature missing rates (%):", nonzero_nulls.to_dict() if not nonzero_nulls.empty else "0.0% across all features")

    last_test_idx = int(df_test_iv.index[-1] + 1)
    df_test_last = df_test[df_test["interval_idx"] == last_test_idx].copy().reset_index(drop=True)
    last_iv_row = df_test_iv.iloc[-1]
    print(
        f"\nBlind Primary Holdout (Last Test Period = Test Int {last_test_idx}): "
        f"[{last_iv_row.start_dt} .. {last_iv_row.end_dt}], n = {len(df_test_last):,}"
    )
    print(f"Blind Full Test Suite (All {len(df_test_iv)} Unique Test Intervals): n = {len(df_test):,}")

    # ------------------------------------------------------------------
    # 1. Leave-One-Interval-Out (GroupKFold) on TRAIN intervals ONLY
    # ------------------------------------------------------------------
    ml_candidates = list(ML_CANDIDATE_NAMES)
    n_train_groups = len(df_train_iv)
    print(f"\n>>> Running {n_train_groups}-Fold Leave-One-Interval-Out (GroupKFold) on TRAIN intervals...")
    gkf = GroupKFold(n_splits=n_train_groups)
    group_cv_summary: Dict[str, Dict[str, Any]] = {}
    group_fold_details: List[Dict[str, Any]] = []

    for mname in ml_candidates:
        tm0 = time.time()
        fold_metrics = []
        for _, (tr_idx, te_idx) in enumerate(gkf.split(df_train, df_train[TARGET_TAG], groups=df_train["interval_idx"])):
            df_tr = df_train.iloc[tr_idx]
            df_te = df_train.iloc[te_idx]
            held_out_int = int(df_te["interval_idx"].iloc[0])
            m_fold, _ = fit_and_eval_split(mname, df_tr, df_te, feature_cols, f"Train_GroupFold_Int{held_out_int}")
            m_fold["held_out_interval"] = held_out_int
            fold_metrics.append(m_fold)
            group_fold_details.append(m_fold)

        group_cv_summary[mname] = {
            "mean_rmse": float(np.mean([x["rmse"] for x in fold_metrics])),
            "mean_mae": float(np.mean([x["mae"] for x in fold_metrics])),
            "mean_mape": float(np.mean([x["mape"] for x in fold_metrics])),
            "mean_accuracy": float(np.mean([x["accuracy"] for x in fold_metrics])),
            "mean_vol_diff_pct": float(np.mean([x["vol_diff_pct"] for x in fold_metrics])),
        }
        s = group_cv_summary[mname]
        print(
            f"  {mname:<16} ({time.time() - tm0:5.1f}s) | {n_train_groups}-Fold GroupKFold Mean: RMSE={s['mean_rmse']:6.2f}, "
            f"MAE={s['mean_mae']:6.2f}, MAPE={s['mean_mape']:5.2f}%, Acc={s['mean_accuracy']:5.2f}%, VolDiff={s['mean_vol_diff_pct']:+5.2f}%"
        )

    # ------------------------------------------------------------------
    # 2. Fit on 100% TRAIN & Blind Evaluation on fuel_valid_test_data.csv
    # ------------------------------------------------------------------
    print("\n>>> Fitting all candidates on 100% TRAIN & evaluating blind on TEST periods...")
    fitted_on_train: Dict[str, Any] = {}
    test_results: List[Dict[str, Any]] = []
    test_last_preds: Dict[str, np.ndarray] = {}
    test_all_preds: Dict[str, np.ndarray] = {}

    for mname in ml_candidates:
        tm0 = time.time()
        est = fit_estimator(build_candidate_models()[mname], df_train, feature_cols)
        fitted_on_train[mname] = est

        m_last, p_last = eval_estimator(est, mname, df_test_last, feature_cols, "Test_LastPeriod")
        m_all, p_all = eval_estimator(est, mname, df_test, feature_cols, "Test_AllPeriods")
        test_results.extend([m_last, m_all])
        test_last_preds[mname] = p_last
        test_all_preds[mname] = p_all
        print(
            f"  {mname:<16} ({time.time() - tm0:5.1f}s) | LastTest RMSE={m_last['rmse']:6.2f} MAPE={m_last['mape']:5.2f}% R2={m_last['r2']:6.4f} "
            f"| AllTest RMSE={m_all['rmse']:6.2f} MAPE={m_all['mape']:5.2f}%"
        )

    # Strict zero-leakage selection purely by unseen GroupKFold CV MAPE/RMSE on TRAIN
    score_board = []
    for mname in ml_candidates:
        gkf_m = group_cv_summary[mname]
        m_last = next(x for x in test_results if x["model"] == mname and x["split"] == "Test_LastPeriod")
        m_all = next(x for x in test_results if x["model"] == mname and x["split"] == "Test_AllPeriods")
        score_board.append({
            "model": mname,
            "train_groupcv_mape": round(gkf_m["mean_mape"], 4),
            "train_groupcv_rmse": round(gkf_m["mean_rmse"], 4),
            "test_last_mape": round(m_last["mape"], 4),
            "test_last_rmse": round(m_last["rmse"], 4),
            "test_last_r2": round(m_last["r2"], 4),
            "test_all_mape": round(m_all["mape"], 4),
            "test_all_rmse": round(m_all["rmse"], 4),
        })
    score_board.sort(key=lambda d: (d["train_groupcv_mape"], d["train_groupcv_rmse"]))
    winner_name = score_board[0]["model"]
    print(
        f"\n>>> SELECTED WINNER MODEL (Zero-Leakage Train GroupKFold): {winner_name} "
        f"(Train CV MAPE={score_board[0]['train_groupcv_mape']:.2f}%, "
        f"Blind LastTest MAPE={score_board[0]['test_last_mape']:.2f}%, "
        f"Blind AllTest MAPE={score_board[0]['test_all_mape']:.2f}%)"
    )

    # ------------------------------------------------------------------
    # 3. 5-Fold Chronological TimeSeriesSplit on TRAIN for Winner
    # ------------------------------------------------------------------
    print(f"\n>>> Running 5-Fold Expanding TimeSeriesSplit on TRAIN for {winner_name}...")
    tscv = TimeSeriesSplit(n_splits=5)
    tscv_records = []
    for fold_idx, (tr_idx, te_idx) in enumerate(tscv.split(df_train)):
        m_ts, _ = fit_and_eval_split(
            winner_name, df_train.iloc[tr_idx], df_train.iloc[te_idx], feature_cols, f"TS_Fold_{fold_idx+1}"
        )
        m_ts["train_start"] = str(df_train.iloc[tr_idx]["Timestamp"].min())[:10]
        m_ts["train_end"] = str(df_train.iloc[tr_idx]["Timestamp"].max())[:10]
        m_ts["test_start"] = str(df_train.iloc[te_idx]["Timestamp"].min())[:10]
        m_ts["test_end"] = str(df_train.iloc[te_idx]["Timestamp"].max())[:10]
        tscv_records.append(m_ts)
        print(
            f"  Fold {fold_idx+1} [{m_ts['test_start']}..{m_ts['test_end']}]: "
            f"RMSE={m_ts['rmse']:6.2f}, MAE={m_ts['mae']:6.2f}, MAPE={m_ts['mape']:5.2f}%, VolDiff={m_ts['vol_diff_pct']:+5.2f}%"
        )

    # ------------------------------------------------------------------
    # 4. Bootstrap CI & Slice Analysis on Primary Blind Test (Last Test Period)
    # ------------------------------------------------------------------
    winner_last_pred = test_last_preds[winner_name]
    winner_all_pred = test_all_preds[winner_name]
    last_test_ci = block_bootstrap_ci(df_test_last[TARGET_TAG].to_numpy(dtype=float), winner_last_pred, block_size=12)
    slice_errors = compute_slice_analysis(df_test_last, winner_last_pred)
    print("\n>>> Blind LastTest 95% Block-Bootstrap CI:", last_test_ci)
    print(">>> Slice-Based Error Analysis (Last Test Period):")
    print(pd.DataFrame(slice_errors)[["slice", "count", "rmse", "mae", "mape", "vol_diff_pct"]].to_string(index=False))

    # ------------------------------------------------------------------
    # 5. Per-Interval Metrics (Train LOIO + Blind Test) & Final Export
    # ------------------------------------------------------------------
    final_model = fitted_on_train[winner_name]
    df_train["F_hat"] = apply_physics_gating(final_model.predict(df_train[feature_cols]), df_train["fuel_gas_active"].to_numpy())
    df_test["F_hat"] = winner_all_pred

    train_interval_records = []
    for idx, row in df_train_iv.iterrows():
        sub = df_train[df_train["interval_idx"] == (idx + 1)]
        m_i = compute_metrics(sub[TARGET_TAG].to_numpy(), sub["F_hat"].to_numpy(), name=f"Train_Int_{idx+1}")
        m_i["interval_idx"] = idx + 1
        m_i["start"] = str(row.start_dt)
        m_i["end"] = str(row.end_dt)
        m_i["mean_actual"] = float(sub[TARGET_TAG].mean())
        m_i["mean_pred"] = float(sub["F_hat"].mean())
        gkf_i = next(x for x in group_fold_details if x["model"] == winner_name and x["held_out_interval"] == (idx + 1))
        m_i["unseen_cv_rmse"] = gkf_i["rmse"]
        m_i["unseen_cv_mape"] = gkf_i["mape"]
        train_interval_records.append(m_i)

    test_interval_records = []
    for idx, row in df_test_iv.iterrows():
        sub = df_test[df_test["interval_idx"] == (idx + 1)]
        m_i = compute_metrics(sub[TARGET_TAG].to_numpy(), sub["F_hat"].to_numpy(), name=f"Test_Int_{idx+1}")
        m_i["interval_idx"] = idx + 1
        m_i["is_last_period"] = bool((idx + 1) == last_test_idx)
        m_i["start"] = str(row.start_dt)
        m_i["end"] = str(row.end_dt)
        m_i["mean_actual"] = float(sub[TARGET_TAG].mean())
        m_i["mean_pred"] = float(sub["F_hat"].mean())
        test_interval_records.append(m_i)

    print("\n>>> Blind Test Per-Interval Summary (Model Never Saw Test Intervals):")
    print(
        pd.DataFrame(test_interval_records)[["interval_idx", "start", "end", "count", "mean_actual", "mean_pred", "rmse", "mape", "vol_diff_pct"]]
        .to_string(index=False)
    )

    feature_importance = extract_feature_importance(final_model, feature_cols, df_test_last)
    print("\nFeature Importance (% on Blind LastTest):")
    print(pd.DataFrame(feature_importance).to_string(index=False))

    meta_weights = getattr(final_model, "meta_weights_", None)
    if meta_weights:
        print("\nGroupKFold StackedEnsemble Meta-Learner Weights:", meta_weights)

    os.makedirs(MODELS_DIR, exist_ok=True)
    winner_last_m = next(x for x in test_results if x["model"] == winner_name and x["split"] == "Test_LastPeriod")
    winner_all_m = next(x for x in test_results if x["model"] == winner_name and x["split"] == "Test_AllPeriods")

    artifact_payload = {
        "model": final_model,
        "model_name": winner_name,
        "target_tag": TARGET_TAG,
        "feature_cols": feature_cols,
        "meta_weights": meta_weights,
        "validation_strategy": (
            f"4-Model Benchmark (StackedEnsemble[GroupKFold], RandomForest, XGBoost, MLP) + "
            f"{n_train_groups}-Fold Train GroupKFold Selection -> Blind Test on fuel_valid_test_data.csv "
            f"(Last Test Period: {last_iv_row.start_dt} .. {last_iv_row.end_dt})"
        ),
        "model_leaderboard": score_board,
        "test_comparison": test_results,
        "group_cv_summary": group_cv_summary,
        "winner_metrics": {
            "test_last_period": winner_last_m,
            "test_last_95ci": last_test_ci,
            "test_all_periods": winner_all_m,
            "train_group_cv_mean": group_cv_summary[winner_name],
            "train_timeseries_cv": tscv_records,
        },
        "last_test_period_info": {
            "interval_idx": last_test_idx,
            "start": str(last_iv_row.start_dt),
            "end": str(last_iv_row.end_dt),
            "count": int(len(df_test_last)),
        },
        "slice_errors_last_test": slice_errors,
        "metrics_train_intervals": train_interval_records,
        "metrics_test_intervals": test_interval_records,
        "feature_importance": feature_importance,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_train_samples": int(len(df_train)),
        "total_test_samples": int(len(df_test)),
        "dropped_null_train_rows": null_train_cnt,
        "dropped_null_test_rows": null_test_cnt,
        "deduplicated_test_intervals": test_dup_cnt,
        "database_source": DB_HISTORIAN_PATH,
        "train_intervals_source": TRAIN_DATA_CSV,
        "test_intervals_source": TEST_DATA_CSV,
    }

    joblib.dump(artifact_payload, MODEL_PATH)
    json_payload = {k: v for k, v in artifact_payload.items() if k != "model"}
    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Summary JSON saved to: {SUMMARY_JSON_PATH}")
    print(f"[COMPLETED] Training workflow finished in {elapsed:.2f}s.")
    return artifact_payload, df_train


if __name__ == "__main__":
    train()
