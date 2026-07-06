import numpy as np
import pandas as pd
import pytest

from risk_dashboard.rolling import rolling_historical_var, rolling_volatility
from risk_dashboard.var import historical_var


def test_rolling_volatility_matches_manual_rolling_std():
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    rng = np.random.default_rng(31)
    returns = pd.Series(rng.normal(0, 0.01, 100), index=idx)

    rolled = rolling_volatility(returns, window=20, annualize=False)
    manual = returns.rolling(20).std()
    pd.testing.assert_series_equal(rolled, manual.rename("rolling_vol"))


def test_rolling_volatility_annualized_scales_by_sqrt_252():
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    returns = pd.Series(np.full(50, 0.01), index=idx)
    raw = rolling_volatility(returns, window=10, annualize=False)
    ann = rolling_volatility(returns, window=10, annualize=True)
    ratio = (ann / raw).dropna()
    assert ratio.unique() == pytest.approx(np.sqrt(252))


def test_rolling_historical_var_matches_pointwise_historical_var():
    idx = pd.date_range("2020-01-01", periods=300, freq="D")
    rng = np.random.default_rng(32)
    returns = pd.Series(rng.normal(0, 0.01, 300), index=idx)

    window = 100
    rolled = rolling_historical_var(returns, window=window, alpha=0.95)
    manual_last = historical_var(returns.iloc[-window:], 0.95)
    assert rolled.iloc[-1] == pytest.approx(manual_last)


def test_rolling_var_increases_when_volatility_regime_shifts_up():
    idx = pd.date_range("2020-01-01", periods=400, freq="D")
    rng = np.random.default_rng(33)
    calm = rng.normal(0, 0.005, 200)
    stressed = rng.normal(0, 0.03, 200)
    returns = pd.Series(np.concatenate([calm, stressed]), index=idx)

    rolled = rolling_historical_var(returns, window=100, alpha=0.95)
    assert rolled.iloc[-1] > rolled.iloc[199]
