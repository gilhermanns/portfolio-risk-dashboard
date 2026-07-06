import numpy as np
import pandas as pd
import pytest

from risk_dashboard.correlation import average_pairwise_correlation, correlation_regime_comparison


def test_average_pairwise_correlation_of_identical_series_is_one():
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    rng = np.random.default_rng(21)
    base = rng.normal(0, 0.01, 100)
    returns = pd.DataFrame({"A": base, "B": base, "C": base}, index=idx)
    assert average_pairwise_correlation(returns) == pytest.approx(1.0)


def test_average_pairwise_correlation_of_independent_series_near_zero():
    idx = pd.date_range("2020-01-01", periods=20000, freq="D")
    rng = np.random.default_rng(22)
    returns = pd.DataFrame(
        {"A": rng.normal(0, 0.01, 20000), "B": rng.normal(0, 0.01, 20000), "C": rng.normal(0, 0.01, 20000)},
        index=idx,
    )
    assert average_pairwise_correlation(returns) == pytest.approx(0.0, abs=0.03)


def test_correlation_regime_detects_higher_crisis_correlation():
    rng = np.random.default_rng(23)
    idx = pd.date_range("2019-01-01", periods=400, freq="D")
    calm_part = rng.normal(0, 0.01, (200, 3))  # independent columns in calm regime

    common_shock = rng.normal(0, 0.02, 200)
    stress_part = common_shock[:, None] + rng.normal(0, 0.002, (200, 3))  # highly correlated in stress

    data = np.vstack([calm_part, stress_part])
    returns = pd.DataFrame(data, index=idx, columns=["A", "B", "C"])

    calm_window = (str(idx[0].date()), str(idx[199].date()))
    stress_windows = {"stress": (str(idx[200].date()), str(idx[-1].date()))}

    regime = correlation_regime_comparison(returns, calm_window, stress_windows)
    assert regime.loc["stress", "avg_pairwise_corr"] > regime.loc["calm", "avg_pairwise_corr"]
