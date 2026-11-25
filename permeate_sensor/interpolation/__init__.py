"""
Interpolation Package for Control Valves.
"""

from .src.valve_spline import (
    ValveSplineInterpolator,
    get_permeate_spline,
    get_fuelgas_spline
)

__all__ = [
    'ValveSplineInterpolator',
    'get_permeate_spline',
    'get_fuelgas_spline'
]
