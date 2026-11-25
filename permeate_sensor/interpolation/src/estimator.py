"""
Two-Stage Estimator for Permeate Flow.
Stage 1: Valve Opening (%) -> Cv (Flow Coefficient) via piecewise linear interpolation
Stage 2: Cv -> Flowrate via linear equation: Q = m * Cv + c (default: Q = k * Cv)
"""

import os
from typing import Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from .config import DEFAULT_PERMEATE_CSV
except ImportError:
    from config import DEFAULT_PERMEATE_CSV


class PermeateFlowEstimator:
    """
    Two-stage permeate flow estimator:
      Stage 1: opening (%) -> Cv using linear interpolation from reference data
      Stage 2: Cv -> flowrate using linear equation: Q = m * Cv + c
    """

    def __init__(
        self,
        csv_path: str = None,
        zero_intercept: bool = True,
        custom_slope: float = None,
        custom_intercept: float = None
    ):
        """
        Initialize the permeate estimator.
        Args:
            csv_path: Path to cv_permeate.csv lookup file
            zero_intercept: If True, forces line through (0, 0) so Q = 0 when Cv = 0 (k * Cv).
            custom_slope: Manually specified slope k/m (optional).
            custom_intercept: Manually specified intercept c (optional).
        """
        self.csv_path = csv_path or DEFAULT_PERMEATE_CSV
        self.zero_intercept = zero_intercept
        self.custom_slope = custom_slope
        self.custom_intercept = custom_intercept

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Lookup table file not found: {self.csv_path}")

        self._load_and_prepare_data()
        self._fit_linear_equation()

    def _load_and_prepare_data(self) -> None:
        """Load and validate reference file cv_permeate.csv."""
        df = pd.read_csv(self.csv_path)

        # Standardize column names to lowercase
        df.columns = [col.strip().lower() for col in df.columns]

        required_cols = {"opening", "cv", "flowrate"}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(
                f"File {self.csv_path} must contain columns {required_cols}. "
                f"Found columns: {list(df.columns)}"
            )

        # Sort ascending by opening
        df = df.sort_values(by="opening").reset_index(drop=True)

        self.table_opening = df["opening"].to_numpy(dtype=float)
        self.table_cv = df["cv"].to_numpy(dtype=float)
        self.table_flow = df["flowrate"].to_numpy(dtype=float)

        self.min_opening = float(self.table_opening[0])
        self.max_opening = float(self.table_opening[-1])
        self.min_cv = float(self.table_cv[0])
        self.max_cv = float(self.table_cv[-1])

    def _fit_linear_equation(self) -> None:
        """Determine linear equation parameters for Stage 2 (Cv -> Flowrate)."""
        if self.custom_slope is not None:
            self.slope = float(self.custom_slope)
            self.intercept = float(self.custom_intercept or 0.0)
        elif self.zero_intercept:
            # Proportional model through origin: Q = k * Cv
            self.slope = float(np.sum(self.table_cv * self.table_flow) / np.sum(self.table_cv ** 2))
            self.intercept = 0.0
        else:
            # Standard OLS regression: Q = m * Cv + c
            self.slope, self.intercept = np.polyfit(self.table_cv, self.table_flow, 1)
            self.slope = float(self.slope)
            self.intercept = float(self.intercept)

    @property
    def linear_equation_str(self) -> str:
        """String representation of Stage 2 linear equation."""
        if abs(self.intercept) < 1e-4:
            return f"Q = {self.slope:.4f} * Cv"
        sign = "+" if self.intercept >= 0 else "-"
        return f"Q = {self.slope:.4f} * Cv {sign} {abs(self.intercept):.4f}"

    def opening_to_cv(self, opening: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray]:
        """Stage 1: Piecewise linear interpolation from valve opening (%) to Cv."""
        arr_opening = np.asarray(opening, dtype=float)
        clamped_opening = np.clip(arr_opening, self.min_opening, self.max_opening)
        calculated_cv = np.interp(clamped_opening, self.table_opening, self.table_cv)

        if np.isscalar(opening):
            return float(calculated_cv)
        return calculated_cv

    def cv_to_flow(self, cv: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray]:
        """Stage 2: Linear equation from Cv to flowrate (Q = m * Cv + c)."""
        arr_cv = np.asarray(cv, dtype=float)
        calculated_flow = self.slope * arr_cv + self.intercept

        # Non-negative clamping when valve is closed
        calculated_flow = np.maximum(calculated_flow, 0.0)

        if np.isscalar(cv):
            return float(calculated_flow)
        return calculated_flow

    def predict(
        self, opening: Union[float, np.ndarray, pd.Series]
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Run both stages sequentially:
        Input:
            opening: Valve opening (%)
        Returns:
            (calculated_cv, estimated_flowrate)
        """
        cv = self.opening_to_cv(opening)
        flow = self.cv_to_flow(cv)
        return cv, flow

    def predict_flow(self, opening: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray]:
        """Return estimated flowrate only."""
        _, flow = self.predict(opening)
        return flow
