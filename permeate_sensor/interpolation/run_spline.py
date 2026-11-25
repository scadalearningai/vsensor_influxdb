"""
Main CLI Entrypoint for Valve Cubic Spline Interpolation Tool.

Usage examples:
  python run_spline.py --valve permeate --opening 45.5
  python run_spline.py --valve fuelgas --flowrate 2500
  python run_spline.py --all
"""

import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from src.valve_spline import (
    ValveSplineInterpolator,
    get_permeate_spline,
    get_fuelgas_spline
)
from src.config import DATA_DIR, FIGURES_DIR


def main():
    parser = argparse.ArgumentParser(description="Generic Valve Cubic Spline Interpolation CLI")
    parser.add_argument('--valve', choices=['permeate', 'fuelgas'], default='permeate',
                        help='Choose predefined valve dataset (default: permeate)')
    parser.add_argument('--file', type=str, default=None,
                        help='Custom CSV path containing opening, cv, flowrate')
    parser.add_argument('--bc', type=str, default='natural', choices=['natural', 'not-a-knot', 'clamped'],
                        help="Spline boundary condition (default: 'natural')")
    parser.add_argument('--opening', type=float, default=None,
                        help='Evaluate flowrate and Cv for a given opening percentage')
    parser.add_argument('--flowrate', type=float, default=None,
                        help='Evaluate required opening percentage for a given target flowrate')
    parser.add_argument('--export', action='store_true',
                        help='Export fine-grid interpolation table to CSV')
    parser.add_argument('--step', type=float, default=0.1,
                        help='Step size for fine-grid export (default: 0.1)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate and save visual plot figure')
    parser.add_argument('--all', action='store_true',
                        help='Run export and plot for both permeate and fuelgas')

    args = parser.parse_args()

    if args.all:
        for valve_key in ['permeate', 'fuelgas']:
            print(f"\n=== Processing Valve: {valve_key.upper()} ===")
            factory = get_permeate_spline if valve_key == 'permeate' else get_fuelgas_spline
            spline = factory(bc_type=args.bc)
            out_csv = os.path.join(DATA_DIR, f"{spline.valve_name}_spline_1000pts.csv")
            spline.save_grid(out_csv, step=args.step)
            spline.plot_curve()
        return

    # Determine file to load
    if args.file:
        csv_path = args.file
        spline = ValveSplineInterpolator(csv_path, bc_type=args.bc)
    elif args.valve == 'fuelgas':
        spline = get_fuelgas_spline(bc_type=args.bc)
    else:
        spline = get_permeate_spline(bc_type=args.bc)

    print(f"\nLoaded Valve Model: {spline.valve_name}")
    print(f"Data Points: {len(spline.openings)} points | Opening Range: {spline.min_opening}% - {spline.max_opening}%")
    print(f"Boundary Condition: {spline.bc_type}")

    # Forward Prediction
    if args.opening is not None:
        flow = spline.predict_flowrate(args.opening)
        cv = spline.predict_cv(args.opening)
        gain = spline.derivative_flowrate(args.opening)
        print(f"\n[Forward Query] Opening: {args.opening:.2f}%")
        print(f"  -> Flowrate : {float(flow):,.2f} lb/h")
        print(f"  -> Cv       : {float(cv):,.2f}")
        print(f"  -> Gain     : {float(gain):,.2f} lb/h per %")

    # Inverse Prediction
    if args.flowrate is not None:
        op = spline.predict_opening(args.flowrate)
        print(f"\n[Inverse Query] Target Flowrate: {args.flowrate:,.2f} lb/h")
        print(f"  -> Required Opening: {float(op):.2f}%")

    # Export grid
    if args.export:
        out_csv = os.path.join(DATA_DIR, f"{spline.valve_name}_spline_1000pts.csv")
        spline.save_grid(out_csv, step=args.step)

    # Plot
    if args.plot:
        spline.plot_curve()

    # Default if no query options given: do both export and plot
    if args.opening is None and args.flowrate is None and not args.export and not args.plot:
        out_csv = os.path.join(DATA_DIR, f"{spline.valve_name}_spline_1000pts.csv")
        spline.save_grid(out_csv, step=args.step)
        spline.plot_curve()


if __name__ == '__main__':
    main()
