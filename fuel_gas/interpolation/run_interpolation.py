"""
Unified CLI Entrypoint for Fuel Gas Control Valve Interpolation Module.

Usage examples:
  python run_interpolation.py                        # Run full linear prediction pipeline -> DuckDB
  python run_interpolation.py --no-save-db --plot    # Dry run + generate figures into reports/figures/
  python run_interpolation.py --opening 22.5         # Query linear piecewise flow & Cv
  python run_interpolation.py --method spline --opening 22.5
  python run_interpolation.py --method spline --flowrate 440
  python run_interpolation.py --method spline --plot # Plot Natural Cubic Spline curve
"""

import argparse
import os
import sys

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from src.config import DEFAULT_CV_PATH, FIGURES_DIR, INFLUX_DATABASE
from src.estimator import ValveFlowEstimator, run_prediction_pipeline
from src.valve_spline import get_fuelgas_spline


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuel Gas Valve Interpolation CLI (Linear & Cubic Spline)")
    parser.add_argument(
        "--method",
        choices=["linear", "spline"],
        default="linear",
        help="Interpolation method (default: linear)",
    )
    parser.add_argument(
        "--bc",
        choices=["natural", "not-a-knot", "clamped"],
        default="natural",
        help="Spline boundary condition when --method spline (default: natural)",
    )
    parser.add_argument("--opening", type=float, default=None, help="Forward query: valve opening (%%) -> Cv & flow")
    parser.add_argument(
        "--flowrate",
        type=float,
        default=None,
        help="Inverse query (spline): target flowrate -> required opening (%%)",
    )
    parser.add_argument("--run-pipeline", action="store_true", help="Run full historian linear pipeline")
    parser.add_argument("--no-save-db", action="store_true", help="Skip saving linear predictions to InfluxDB")
    parser.add_argument("--plot", action="store_true", help="Generate visual plots into interpolation/reports/figures/")
    args = parser.parse_args()

    if args.method == "spline":
        spline = get_fuelgas_spline(bc_type=args.bc)
        if args.opening is not None:
            flow = float(spline.predict_flowrate(args.opening))
            cv = float(spline.predict_cv(args.opening))
            gain = float(spline.derivative_flowrate(args.opening))
            print(f"[Spline Forward] Opening: {args.opening:.2f}% -> Flow: {flow:,.2f} | Cv: {cv:,.2f} | Gain: {gain:,.2f}/%")
        if args.flowrate is not None:
            op = float(spline.predict_opening(args.flowrate))
            print(f"[Spline Inverse] Target Flow: {args.flowrate:,.2f} -> Required Opening: {op:.2f}%")
        if args.plot or (args.opening is None and args.flowrate is None):
            out_img = os.path.join(FIGURES_DIR, "cv_fuelgas_spline_curve.png")
            spline.plot_curve(out_img)
        return

    # Piecewise linear method
    est = ValveFlowEstimator(DEFAULT_CV_PATH)
    has_query = args.opening is not None or args.flowrate is not None

    if args.opening is not None:
        cv = float(est.opening_to_cv(args.opening))
        flow = float(est.predict_flow(args.opening))
        print(f"[Linear Forward] Opening: {args.opening:.2f}% -> Cv: {cv:.2f} | Flow: {flow:,.2f}")

    if args.flowrate is not None:
        # Exact piecewise inversion on tabulated flowrate points
        inv_op = float(
            est.cv_df["opening"].to_numpy()[0]
            if args.flowrate <= 0
            else __import__("numpy").interp(
                args.flowrate,
                (est.cv_values * est.cv_to_flow_factor),
                est.openings,
            )
        )
        print(f"[Linear Inverse] Target Flow: {args.flowrate:,.2f} -> Estimated Opening: {inv_op:.2f}%")

    if args.run_pipeline or not has_query:
        run_prediction_pipeline(
            database=INFLUX_DATABASE,
            cv_csv_path=DEFAULT_CV_PATH,
            save_to_db=not args.no_save_db,
        )

    if args.plot:
        from scripts.plot_prediction_results import (
            create_prediction_overview_plot,
            create_valve_characteristic_plot,
        )

        create_valve_characteristic_plot(est)
        create_prediction_overview_plot(INFLUX_DATABASE)


if __name__ == "__main__":
    main()

