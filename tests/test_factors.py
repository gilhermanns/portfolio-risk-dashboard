import numpy as np
import pandas as pd
import pytest

from risk_dashboard.factors import factor_regression


def test_single_factor_regression_recovers_true_beta():
    rng = np.random.default_rng(11)
    n = 5000
    factor = pd.Series(rng.normal(0.0002, 0.01, n), name="MKT")
    true_beta = 1.35
    true_alpha = 0.0001
    noise = rng.normal(0.0, 0.002, n)
    portfolio = true_alpha + true_beta * factor + noise
    portfolio = pd.Series(portfolio, name="portfolio")

    result = factor_regression(portfolio, pd.DataFrame({"MKT": factor}))

    assert result.betas["MKT"] == pytest.approx(true_beta, abs=0.03)
    assert result.alpha == pytest.approx(true_alpha, abs=0.0005)
    assert result.r_squared > 0.9


def test_multi_factor_regression_recovers_betas_and_variance_explained():
    rng = np.random.default_rng(12)
    n = 8000
    mkt = rng.normal(0.0003, 0.011, n)
    rates = rng.normal(0.0001, 0.004, n)
    # Uncorrelated factors so the "ignore cross-covariance" variance decomposition is exact.
    factors_df = pd.DataFrame({"MKT": mkt, "RATES": rates})

    beta_mkt, beta_rates = 0.8, -0.5
    noise = rng.normal(0.0, 0.001, n)
    portfolio = pd.Series(beta_mkt * mkt + beta_rates * rates + noise)

    result = factor_regression(portfolio, factors_df)

    assert result.betas["MKT"] == pytest.approx(beta_mkt, abs=0.02)
    assert result.betas["RATES"] == pytest.approx(beta_rates, abs=0.02)
    assert sum(result.variance_explained.values()) == pytest.approx(result.r_squared, abs=0.03)
    assert result.residual_variance_share == pytest.approx(1 - result.r_squared, abs=0.03)


def test_zero_beta_when_portfolio_is_pure_noise():
    rng = np.random.default_rng(13)
    n = 3000
    factor = pd.Series(rng.normal(0, 0.01, n), name="MKT")
    portfolio = pd.Series(rng.normal(0, 0.01, n))

    result = factor_regression(portfolio, pd.DataFrame({"MKT": factor}))
    assert result.betas["MKT"] == pytest.approx(0.0, abs=0.05)
    assert result.r_squared < 0.05
