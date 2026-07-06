import numpy as np
import pandas as pd
import pytest

from risk_dashboard.portfolio import (
    herfindahl_index,
    portfolio_returns,
    renormalized_weights,
    simple_returns,
)


def test_herfindahl_equal_weight_n_assets():
    weights = {f"A{i}": 1 / 10 for i in range(10)}
    assert herfindahl_index(weights) == pytest.approx(1 / 10)


def test_herfindahl_full_concentration():
    weights = {"A": 1.0}
    assert herfindahl_index(weights) == pytest.approx(1.0)


def test_herfindahl_between_equal_and_concentrated():
    equal = herfindahl_index({"A": 0.5, "B": 0.5})
    concentrated = herfindahl_index({"A": 0.9, "B": 0.1})
    assert concentrated > equal


def test_renormalized_weights_sums_to_one_and_preserves_ratio():
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    sub = renormalized_weights(weights, ["A", "B"])
    assert sub["A"] + sub["B"] == pytest.approx(1.0)
    assert sub["A"] / sub["B"] == pytest.approx(0.5 / 0.3)


def test_portfolio_returns_matches_manual_weighted_sum():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    returns = pd.DataFrame({"A": [0.01, -0.02, 0.03, 0.0, 0.01], "B": [0.02, 0.01, -0.01, 0.02, -0.02]}, index=idx)
    weights = {"A": 0.4, "B": 0.6}
    got = portfolio_returns(returns, weights)
    expected = returns["A"] * 0.4 + returns["B"] * 0.6
    pd.testing.assert_series_equal(got, expected.rename("portfolio"), check_dtype=False)


def test_simple_returns_matches_pct_change():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    levels = pd.DataFrame({"A": [100, 110, 99, 108.9]}, index=idx)
    returns = simple_returns(levels)
    assert returns["A"].tolist() == pytest.approx([0.10, -0.10, 0.1])
