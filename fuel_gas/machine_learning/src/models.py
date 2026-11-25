"""
Model Definitions for Fuel Gas Soft Sensor (46-FIC-103A1.A).

Implements the 4 candidate architectures (SVR excluded per instruction):
1. StackedEnsemble (GroupKFold out-of-fold stacking of RandomForest, XGBoost, MLP + RidgeCV meta)
2. RandomForest (RandomForestRegressor with median imputation)
3. XGBoost (XGBRegressor with native histogram splits & missing-value handling)
4. MLP (Multi-Layer Perceptron neural network with median imputation + standard scaling)
"""

from typing import Callable, Dict, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
import xgboost as xgb

from src.config import RANDOM_SEED

ML_CANDIDATE_NAMES: List[str] = [
    "StackedEnsemble",
    "RandomForest",
    "XGBoost",
    "MLP",
]


def _to_float_array(X) -> np.ndarray:
    """Normalizes DataFrame or array-like inputs to float64 numpy arrays for consistent pipelines."""
    return np.asarray(X, dtype=float)


class GroupStackedRegressor(BaseEstimator, RegressorMixin):
    """
    Out-of-fold Stacking Regressor built on GroupKFold by operational interval (`interval_idx`)
    so meta-learner weights never overfit adjacent timestamp autocorrelation.
    """

    def __init__(
        self,
        estimator_factory: Callable[[], Dict[str, BaseEstimator]],
        n_splits: int = 8,
    ):
        self.estimator_factory = estimator_factory
        self.n_splits = n_splits

    def fit(self, X, y, groups: Optional[np.ndarray] = None):
        X_arr = _to_float_array(X)
        y_arr = np.asarray(y, dtype=float)
        proto = self.estimator_factory()
        self.base_names_ = list(proto.keys())
        n_base = len(self.base_names_)

        if groups is not None and len(np.unique(groups)) >= 2:
            n_unique = len(np.unique(groups))
            splitter = GroupKFold(n_splits=min(self.n_splits, n_unique))
            splits = list(splitter.split(X_arr, y_arr, groups=groups))
        else:
            splitter = KFold(n_splits=min(self.n_splits, max(2, len(X_arr) // 500)), shuffle=False)
            splits = list(splitter.split(X_arr, y_arr))

        oof_preds = np.zeros((len(X_arr), n_base), dtype=float)
        for tr_idx, val_idx in splits:
            fold_models = self.estimator_factory()
            for j, name in enumerate(self.base_names_):
                est = fold_models[name]
                est.fit(X_arr[tr_idx], y_arr[tr_idx])
                oof_preds[val_idx, j] = est.predict(X_arr[val_idx])

        self.meta_learner_ = RidgeCV(alphas=np.logspace(-1, 4, 12))
        self.meta_learner_.fit(oof_preds, y_arr)
        self.meta_weights_ = {
            name: round(float(coef), 4)
            for name, coef in zip(self.base_names_, self.meta_learner_.coef_)
        }

        self.fitted_bases_: Dict[str, BaseEstimator] = {}
        final_models = self.estimator_factory()
        for name in self.base_names_:
            est = final_models[name]
            est.fit(X_arr, y_arr)
            self.fitted_bases_[name] = est

        return self

    def predict(self, X) -> np.ndarray:
        X_arr = _to_float_array(X)
        meta_features = np.column_stack([
            self.fitted_bases_[name].predict(X_arr) for name in self.base_names_
        ])
        return self.meta_learner_.predict(meta_features)


class _ArrayPipeline(BaseEstimator, RegressorMixin):
    """Normalizes inputs to ndarray before delegating to an sklearn Pipeline or estimator."""

    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(_to_float_array(X), np.asarray(y, dtype=float))
        return self

    def predict(self, X) -> np.ndarray:
        return self.estimator.predict(_to_float_array(X))


def _build_base_models() -> Dict[str, BaseEstimator]:
    """Single source of truth for base candidate hyperparameters (shared with StackedEnsemble)."""
    return {
        "RandomForest": _ArrayPipeline(
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=120,
                        max_depth=14,
                        min_samples_leaf=10,
                        max_samples=0.6,
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ])
        ),
        "XGBoost": _ArrayPipeline(
            xgb.XGBRegressor(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=2.0,
                min_child_weight=10,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                tree_method="hist",
            )
        ),
        "MLP": _ArrayPipeline(
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=1e-3,
                        batch_size=1024,
                        learning_rate_init=0.003,
                        max_iter=60,
                        tol=1e-2,
                        n_iter_no_change=5,
                        early_stopping=True,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ])
        ),
    }


def build_candidate_models() -> Dict[str, BaseEstimator]:
    """Returns fresh instances of the 5 benchmark candidate models."""
    models: Dict[str, BaseEstimator] = {
        "StackedEnsemble": GroupStackedRegressor(estimator_factory=_build_base_models, n_splits=5),
    }
    models.update(_build_base_models())
    return models
