"""Value-at-Risk and Expected Shortfall via three methods: historical simulation,
parametric (variance-covariance, Normal and Student-t), and Monte Carlo.

All VaR/ES figures are returned as positive numbers representing a magnitude of
loss over one day at the given confidence level (e.g. VaR=0.02 means a 2% loss).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def historical_var(returns: pd.Series | np.ndarray, alpha: float = 0.95) -> float:
    q = 1.0 - alpha
    return float(-np.quantile(np.asarray(returns), q))


def historical_es(returns: pd.Series | np.ndarray, alpha: float = 0.95) -> float:
    returns = np.asarray(returns)
    q = 1.0 - alpha
    threshold = np.quantile(returns, q)
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        tail = np.array([threshold])
    return float(-tail.mean())


def parametric_var_normal(returns: pd.Series | np.ndarray, alpha: float = 0.95) -> float:
    returns = np.asarray(returns)
    mu, sigma = returns.mean(), returns.std(ddof=1)
    z = stats.norm.ppf(1.0 - alpha)
    return float(-(mu + sigma * z))


def parametric_es_normal(returns: pd.Series | np.ndarray, alpha: float = 0.95) -> float:
    returns = np.asarray(returns)
    mu, sigma = returns.mean(), returns.std(ddof=1)
    q = 1.0 - alpha
    z = stats.norm.ppf(q)
    es = mu - sigma * stats.norm.pdf(z) / q
    return float(-es)


def _t_scale_from_std(sigma: float, dof: float) -> float:
    """Scale parameter of a Student-t with `dof` degrees of freedom whose variance
    matches the sample variance sigma**2 (variance of standard t is dof/(dof-2))."""
    return float(sigma * np.sqrt((dof - 2.0) / dof))


def parametric_var_t(returns: pd.Series | np.ndarray, alpha: float = 0.95, dof: float = 5.0) -> float:
    returns = np.asarray(returns)
    mu, sigma = returns.mean(), returns.std(ddof=1)
    scale = _t_scale_from_std(sigma, dof)
    t_q = stats.t.ppf(1.0 - alpha, dof)
    return float(-(mu + scale * t_q))


def parametric_es_t(returns: pd.Series | np.ndarray, alpha: float = 0.95, dof: float = 5.0) -> float:
    returns = np.asarray(returns)
    mu, sigma = returns.mean(), returns.std(ddof=1)
    scale = _t_scale_from_std(sigma, dof)
    q = 1.0 - alpha
    t_q = stats.t.ppf(q, dof)
    es_standard = (dof + t_q**2) / (dof - 1.0) * stats.t.pdf(t_q, dof) / q
    return float(-(mu - scale * es_standard))


def monte_carlo_var_es(
    asset_returns: pd.DataFrame,
    weights: dict[str, float],
    alpha: float = 0.95,
    n_sims: int = 20000,
    seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """Simulate correlated daily asset returns via a Cholesky decomposition of the
    sample covariance matrix (equivalent to one discrete step of correlated GBM on
    log-returns), weight into portfolio returns, and read VaR/ES off the simulated
    distribution.
    """
    cols = [c for c in asset_returns.columns if c in weights]
    mu = asset_returns[cols].mean().to_numpy()
    cov = asset_returns[cols].cov().to_numpy()
    cov = cov + np.eye(len(cols)) * 1e-14  # numerical floor for near-singular covariances
    L = np.linalg.cholesky(cov)

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_sims, len(cols)))
    sim_asset_returns = mu + z @ L.T

    w = np.array([weights[c] for c in cols])
    sim_port = sim_asset_returns @ w

    var = historical_var(sim_port, alpha)
    es = historical_es(sim_port, alpha)
    return var, es, sim_port


def var_es_table(
    returns: pd.Series,
    alpha_levels: tuple[float, ...] = (0.95, 0.99),
    t_dof: float = 5.0,
    asset_returns: pd.DataFrame | None = None,
    weights: dict[str, float] | None = None,
    n_sims: int = 20000,
) -> pd.DataFrame:
    rows = []
    for alpha in alpha_levels:
        row = {
            "confidence": alpha,
            "historical_VaR": historical_var(returns, alpha),
            "historical_ES": historical_es(returns, alpha),
            "parametric_normal_VaR": parametric_var_normal(returns, alpha),
            "parametric_normal_ES": parametric_es_normal(returns, alpha),
            "parametric_t_VaR": parametric_var_t(returns, alpha, t_dof),
            "parametric_t_ES": parametric_es_t(returns, alpha, t_dof),
        }
        if asset_returns is not None and weights is not None:
            mc_var, mc_es, _ = monte_carlo_var_es(asset_returns, weights, alpha, n_sims=n_sims)
            row["monte_carlo_VaR"] = mc_var
            row["monte_carlo_ES"] = mc_es
        rows.append(row)
    return pd.DataFrame(rows).set_index("confidence")
