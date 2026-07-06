from __future__ import annotations

import numpy as np
import pandas as pd


def simple_returns(levels: pd.DataFrame) -> pd.DataFrame:
    return levels.pct_change().dropna(how="all")


def renormalized_weights(weights: dict[str, float], available: list[str]) -> dict[str, float]:
    """Rescale weights to sum to 1 over only the assets present in `available`,
    preserving their relative proportions. Used when a scenario window only has
    data for a subset of the configured portfolio (e.g. factor proxies but not
    individual stocks before their price history begins).
    """
    sub = {k: v for k, v in weights.items() if k in available}
    total = sum(sub.values())
    if total <= 0:
        raise ValueError("No overlapping weight for the available assets")
    return {k: v / total for k, v in sub.items()}


def portfolio_returns(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    cols = [c for c in returns.columns if c in weights]
    w = np.array([weights[c] for c in cols])
    sub = returns[cols].dropna(how="any")
    port = sub.to_numpy() @ w
    return pd.Series(port, index=sub.index, name="portfolio")


def herfindahl_index(weights: dict[str, float]) -> float:
    """Sum of squared portfolio weights; 1/n for an equally-weighted n-asset book,
    1.0 for full concentration in a single asset.
    """
    return float(sum(w**2 for w in weights.values()))
