"""
Core Valve Cubic Spline Interpolation Module for Fuel Gas.
Fits smooth C2-continuous splines for:
  - opening (%) -> flowrate
  - opening (%) -> Cv
  - flowrate -> opening (%) (inverse)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

try:
    from .config import DEFAULT_CV_PATH, FIGURES_DIR
except ImportError:
    from config import DEFAULT_CV_PATH, FIGURES_DIR


class ValveSplineInterpolator:
    """Generic Cubic Spline Interpolator for control valves."""

    def __init__(self, csv_path: str = DEFAULT_CV_PATH, bc_type: str = "natural", valve_name: str = None):
        self.csv_path = csv_path
        self.bc_type = bc_type
        self.valve_name = valve_name or os.path.splitext(os.path.basename(csv_path))[0]

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV data file not found: {csv_path}")

        self.df_raw = pd.read_csv(csv_path)
        self.df_raw.columns = [col.strip().lower() for col in self.df_raw.columns]

        required_cols = {"opening", "cv", "flowrate"}
        if not required_cols.issubset(set(self.df_raw.columns)):
            raise ValueError(
                f"File {csv_path} must contain columns {required_cols}. "
                f"Found columns: {list(self.df_raw.columns)}"
            )

        self.df_raw = self.df_raw.sort_values(by="opening").reset_index(drop=True)
        self.openings = self.df_raw["opening"].to_numpy(dtype=float)
        self.cvs = self.df_raw["cv"].to_numpy(dtype=float)
        self.flowrates = self.df_raw["flowrate"].to_numpy(dtype=float)

        self.min_opening = float(self.openings[0])
        self.max_opening = float(self.openings[-1])
        self.min_flow = float(self.flowrates[0])
        self.max_flow = float(self.flowrates[-1])

        self.spline_flow = CubicSpline(self.openings, self.flowrates, bc_type=self.bc_type)
        self.spline_cv = CubicSpline(self.openings, self.cvs, bc_type=self.bc_type)

        if np.all(np.diff(self.flowrates) > 0):
            self.spline_opening = CubicSpline(self.flowrates, self.openings, bc_type=self.bc_type)
        else:
            self.spline_opening = None

    def predict_flowrate(self, opening):
        """Estimate flowrate for given opening (%)"""
        op = np.asarray(opening, dtype=float)
        return self.spline_flow(np.clip(op, self.min_opening, self.max_opening))

    def predict_cv(self, opening):
        """Estimate Cv for given opening (%)"""
        op = np.asarray(opening, dtype=float)
        return self.spline_cv(np.clip(op, self.min_opening, self.max_opening))

    def predict_opening(self, flowrate):
        """Estimate opening (%) from target flowrate (Inverse Spline)"""
        if self.spline_opening is None:
            raise ValueError("Flowrate data is not strictly monotonically increasing; inverse spline is unavailable.")
        fl = np.asarray(flowrate, dtype=float)
        return self.spline_opening(np.clip(fl, self.min_flow, self.max_flow))

    def derivative_flowrate(self, opening):
        """First derivative d(flowrate)/d(opening) showing valve sensitivity / gain"""
        op = np.asarray(opening, dtype=float)
        return self.spline_flow.derivative(1)(np.clip(op, self.min_opening, self.max_opening))

    def generate_grid(self, step: float = 0.1) -> pd.DataFrame:
        """Generate fine-grid interpolation table."""
        num_points = int(round((self.max_opening - self.min_opening) / step)) + 1
        grid_openings = np.linspace(self.min_opening, self.max_opening, num_points)
        grid_cv = self.predict_cv(grid_openings)
        grid_flow = self.predict_flowrate(grid_openings)
        grid_df = self.derivative_flowrate(grid_openings)

        if self.min_flow >= 0:
            grid_flow = np.maximum(grid_flow, 0.0)
        if self.cvs[0] >= 0:
            grid_cv = np.maximum(grid_cv, 0.0)

        return pd.DataFrame({
            "opening": np.round(grid_openings, 4),
            "cv_spline": np.round(grid_cv, 4),
            "flowrate_spline": np.round(grid_flow, 4),
            "gain_df_dop": np.round(grid_df, 4),
        })

    def save_grid(self, output_csv_path: str, step: float = 0.1) -> str:
        """Export fine-grid table to CSV."""
        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        df_grid = self.generate_grid(step=step)
        df_grid.to_csv(output_csv_path, index=False)
        print(f"[{self.valve_name}] Fine-grid table saved to: {output_csv_path} ({len(df_grid)} rows)")
        return output_csv_path

    def plot_curve(self, output_img_path: str = None, show: bool = False) -> str:
        """Generate high-resolution Cubic Spline curve visualization."""
        if output_img_path is None:
            os.makedirs(FIGURES_DIR, exist_ok=True)
            output_img_path = os.path.join(FIGURES_DIR, f"{self.valve_name}_spline_curve.png")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(output_img_path)), exist_ok=True)

        x_dense = np.linspace(self.min_opening, self.max_opening, 1000)
        y_flow_dense = self.predict_flowrate(x_dense)
        y_cv_dense = self.predict_cv(x_dense)
        y_gain_dense = self.derivative_flowrate(x_dense)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        ax1.plot(x_dense, y_flow_dense, color="#007acc", lw=2.2, label=f"Cubic Spline (bc={self.bc_type})")
        ax1.scatter(self.openings, self.flowrates, color="#d9534f", s=60, zorder=5, label="Actual Data Points", edgecolor="black")
        ax1.set_ylabel("Flowrate", fontsize=11, fontweight="bold")
        ax1.set_title(f"Valve Characterization: {self.valve_name.upper()} (Natural Cubic Spline)", fontsize=14, fontweight="bold", pad=12)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(loc="upper left", frameon=True)

        ax2.plot(x_dense, y_cv_dense, color="#28a745", lw=2.2, label="Cv Spline")
        ax2.scatter(self.openings, self.cvs, color="#ffc107", s=60, zorder=5, label="Actual Cv Points", edgecolor="black")
        ax2.set_ylabel("Valve Cv", fontsize=11, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="upper left", frameon=True)

        ax3.plot(x_dense, y_gain_dense, color="#6f42c1", lw=2.0, label="Valve Gain (dFlow/dOpening)")
        ax3.set_xlabel("Valve Opening (%)", fontsize=11, fontweight="bold")
        ax3.set_ylabel("Gain (per %)", fontsize=11, fontweight="bold")
        ax3.grid(True, linestyle="--", alpha=0.6)
        ax3.legend(loc="upper left", frameon=True)

        plt.tight_layout()
        plt.savefig(output_img_path, dpi=300)
        print(f"[{self.valve_name}] Curve plot saved to: {output_img_path}")
        if show:
            plt.show()
        plt.close(fig)
        return output_img_path


def get_fuelgas_spline(bc_type: str = "natural") -> ValveSplineInterpolator:
    """Factory helper to obtain fuel gas control valve (46-FV-103A) spline model."""
    return ValveSplineInterpolator(DEFAULT_CV_PATH, bc_type=bc_type, valve_name="cv_fuelgas")

