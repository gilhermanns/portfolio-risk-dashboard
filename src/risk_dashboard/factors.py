from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FactorRegressionResult:
    alpha: float
    betas: dict[str, float]
    r_squared: float
    variance_explained: dict[str, float]  # fraction of portfolio variance attributed to each factor
    residual_variance_share: float


def factor_regression(portfolio_returns: pd.Series, factor_returns: pd.DataFrame) -> FactorRegressionResult:
    """OLS regression of portfolio returns on factor returns via least squares.

    Variance attribution per factor uses the standard "ignore cross-covariance"
    approximation: beta_i**2 * var(factor_i) / var(portfolio), which is exact only
    for uncorrelated factors; correlated factors will not sum exactly to R-squared
    and the gap is reported separately.
    """
    df = pd.concat([portfolio_returns.rename("portfolio"), factor_returns], axis=1).dropna()
    y = df["portfolio"].to_numpy()
    factor_cols = list(factor_returns.columns)
    X = df[factor_cols].to_numpy()
    X_design = np.column_stack([np.ones(len(X)), X])

    coefs, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    alpha = float(coefs[0])
    betas = {col: float(b) for col, b in zip(factor_cols, coefs[1:])}

    y_hat = X_design @ coefs
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    port_var = float(np.var(y))
    variance_explained = {}
    for col in factor_cols:
        factor_var = float(np.var(df[col].to_numpy()))
        variance_explained[col] = (betas[col] ** 2 * factor_var) / port_var if port_var > 0 else 0.0
    residual_variance_share = max(0.0, 1.0 - sum(variance_explained.values()))

    return FactorRegressionResult(
        alpha=alpha,
        betas=betas,
        r_squared=r_squared,
        variance_explained=variance_explained,
        residual_variance_share=residual_variance_share,
    )
