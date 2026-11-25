"""
Fuel Gas Machine Learning Soft Sensor Package (46-FIC-103A1.A).
"""

from .src.inference import FuelGasSoftSensorEngine, run_inference

__all__ = [
    "FuelGasSoftSensorEngine",
    "run_inference",
]

