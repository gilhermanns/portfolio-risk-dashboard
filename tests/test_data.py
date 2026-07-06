import numpy as np
import pandas as pd
import pytest

from risk_dashboard.data import bond_return_from_yield, build_price_levels, common_window


def test_bond_return_from_yield_carry_only_when_yield_flat():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    flat_yield = pd.Series([4.0, 4.0, 4.0, 4.0, 4.0], index=idx)
    r = bond_return_from_yield(flat_yield, duration=7.5)
    # No yield change -> return should equal pure daily carry (previous day's yield / 252)
    assert r.iloc[2] == pytest.approx(0.04 / 252)


def test_bond_return_from_yield_falls_when_yields_rise():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    rising_yield = pd.Series([4.0, 4.5, 4.5], index=idx)
    r = bond_return_from_yield(rising_yield, duration=7.5)
    # A yield increase should produce a negative price-effect large enough to dominate carry.
    assert r.iloc[1] < 0


def test_build_price_levels_offline_has_expected_columns():
    levels = build_price_levels(use_live=False)
    expected_cols = {
        "MKT_NASDAQCOM",
        "GROWTH_NASDAQ100",
        "BOND_DGS10",
        "OIL_WTI",
        "AAPL",
        "XOM",
        "JPM",
        "PFE",
        "WMT",
        "GE",
        "BAC",
        "T",
    }
    assert expected_cols.issubset(set(levels.columns))
    assert len(levels) > 5000


def test_common_window_has_no_missing_values():
    levels = build_price_levels(use_live=False)
    cols = ["MKT_NASDAQCOM", "BOND_DGS10", "AAPL", "JPM"]
    window = common_window(levels, cols)
    assert window.isna().sum().sum() == 0
    assert len(window) > 1000
