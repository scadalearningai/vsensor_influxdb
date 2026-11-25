"""
Unit Tests for Permeate Flow Two-Stage Estimator.
Stage 1: Opening -> Cv (Interpolation)
Stage 2: Cv -> Flow (Linear Equation Q = k * Cv)
"""

import numpy as np
import pytest

from .config import DEFAULT_PERMEATE_CSV
from .estimator import PermeateFlowEstimator


@pytest.fixture
def estimator():
    return PermeateFlowEstimator(DEFAULT_PERMEATE_CSV)


def test_stage1_cv_exact(estimator):
    """Ensure CSV valve opening points yield exact Cv values."""
    for open_val, expected_cv in zip(estimator.table_opening, estimator.table_cv):
        cv = estimator.opening_to_cv(open_val)
        assert np.isclose(cv, expected_cv, atol=1e-5), f"Cv mismatch at opening {open_val}"


def test_stage2_linear_equation(estimator):
    """Ensure Stage 2 uses linear equation Q = k * Cv."""
    for cv_val in [0.0, 100.0, 1500.0, 5000.0, 12338.0]:
        expected_flow = estimator.slope * cv_val + estimator.intercept
        actual_flow = estimator.cv_to_flow(cv_val)
        assert np.isclose(actual_flow, expected_flow, atol=1e-5)

    # Verify consistency with CSV table (< 0.1% relative deviation from linear curve)
    for cv_val, table_flow in zip(estimator.table_cv[1:], estimator.table_flow[1:]):
        flow = estimator.cv_to_flow(cv_val)
        rel_diff = abs(flow - table_flow) / table_flow
        assert rel_diff < 0.001, f"Linear approximation deviation too large at Cv={cv_val}"


def test_clamping_boundary(estimator):
    """Test that out-of-range inputs (opening < 0% or > 100%) are safely clamped."""
    cv_neg, flow_neg = estimator.predict(-15.0)
    assert np.isclose(cv_neg, estimator.table_cv[0])
    assert np.isclose(flow_neg, 0.0)

    cv_over, flow_over = estimator.predict(120.0)
    assert np.isclose(cv_over, estimator.table_cv[-1])
    assert np.isclose(flow_over, estimator.cv_to_flow(estimator.table_cv[-1]))


def test_vectorized_prediction(estimator):
    """Test that NumPy array / list inputs are processed properly."""
    test_openings = np.array([0.0, 11.11, 22.22, 50.0, 100.0])
    cvs, flows = estimator.predict(test_openings)
    assert len(cvs) == len(test_openings)
    assert len(flows) == len(test_openings)
    assert np.all(np.diff(cvs) >= 0)
    assert np.all(np.diff(flows) >= 0)


if __name__ == "__main__":
    est = PermeateFlowEstimator()
    test_stage1_cv_exact(est)
    test_stage2_linear_equation(est)
    test_clamping_boundary(est)
    test_vectorized_prediction(est)
    print("All PermeateFlowEstimator unit tests (Stage 1 Interp + Stage 2 Linear) PASSED (100%)!")
