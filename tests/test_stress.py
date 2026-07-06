import numpy as np
import pandas as pd
import pytest

from risk_dashboard.portfolio import portfolio_returns
from risk_dashboard.stress import apply_historical_scenario, max_drawdown, worst_n_day_drawdown


def test_worst_n_day_drawdown_finds_known_worst_window():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    # A single catastrophic 2-day drop is embedded at a known location.
    values = [0.001] * 4 + [-0.10, -0.08] + [0.001] * 4
    returns = pd.Series(values, index=idx)

    worst, start, end = worst_n_day_drawdown(returns, n=2)
    cum = (1 + returns).cumprod()
    expected = cum.iloc[5] / cum.iloc[3] - 1
    assert worst == pytest.approx(expected)
    assert end == idx[5]
    assert start == idx[3]


def test_worst_n_day_drawdown_is_worse_than_any_single_day_for_a_sustained_decline():
    idx = pd.date_range("2020-01-01", periods=20, freq="D")
    returns = pd.Series([-0.01] * 20, index=idx)
    worst_1, _, _ = worst_n_day_drawdown(returns, n=1)
    worst_10, _, _ = worst_n_day_drawdown(returns, n=10)
    assert worst_10 < worst_1  # a sustained 10-day decline compounds to something worse than one day


def test_max_drawdown_recovers_known_value():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    # Grows to 1.10, falls to 0.88 -> drawdown of 0.88/1.10 - 1
    returns = pd.Series([0.10, 0.0, -0.10, -0.10, 0.05], index=idx)
    dd = max_drawdown(returns)
    cum = (1 + returns).cumprod()
    expected = float((cum / cum.cummax() - 1).min())
    assert dd == pytest.approx(expected)


def test_apply_historical_scenario_renormalizes_when_asset_missing():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    returns = pd.DataFrame(
        {
            "A": [0.01, -0.02, 0.03, 0.0, 0.01],
            "B": [np.nan] * 5,  # never available in this scenario window
        },
        index=idx,
    )
    weights = {"A": 0.5, "B": 0.5}
    result = apply_historical_scenario(returns, weights, "test_scenario", "2020-01-01", "2020-01-05")

    assert result.assets_used == ["A"]
    assert result.weight_coverage == pytest.approx(0.5)
    expected_cum = float(np.prod(1 + returns["A"].to_numpy()) - 1)
    assert result.cumulative_return == pytest.approx(expected_cum)


def test_apply_historical_scenario_full_coverage_matches_direct_calculation():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    returns = pd.DataFrame(
        {"A": [0.01, -0.02, 0.03, 0.0, 0.01], "B": [0.02, 0.01, -0.01, 0.02, -0.02]},
        index=idx,
    )
    weights = {"A": 0.5, "B": 0.5}
    result = apply_historical_scenario(returns, weights, "full_coverage", "2020-01-01", "2020-01-05")

    assert result.weight_coverage == pytest.approx(1.0)
    port = portfolio_returns(returns, weights)
    expected_cum = float(np.prod(1 + port.to_numpy()) - 1)
    assert result.cumulative_return == pytest.approx(expected_cum)
