from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from risk_dashboard.portfolio import renormalized_weights


@dataclass
class ScenarioResult:
    name: str
    start: str
    end: str
    n_days: int
    assets_used: list[str]
    weight_coverage: float  # fraction of the configured portfolio's weight represented
    cumulative_return: float
    worst_day_return: float


def apply_historical_scenario(
    returns: pd.DataFrame, weights: dict[str, float], name: str, start: str, end: str
) -> ScenarioResult:
    """Replay a real historical return sequence (e.g. the 2008 crash) against the
    *current* portfolio weights. If some configured assets have no data in the
    window (e.g. individual stocks before their price history begins), weights are
    renormalized over the assets that do, and the covered weight share is reported
    so the result's completeness is transparent.
    """
    window = returns.loc[start:end]
    available = [c for c in window.columns if window[c].notna().all() and c in weights]
    window = window[available].dropna(how="any")
    w = renormalized_weights(weights, available)
    coverage = sum(weights[a] for a in available)

    port = window.to_numpy() @ np.array([w[c] for c in available])
    cumulative = float(np.prod(1.0 + port) - 1.0)
    worst_day = float(port.min()) if len(port) else float("nan")

    return ScenarioResult(
        name=name,
        start=start,
        end=end,
        n_days=len(window),
        assets_used=available,
        weight_coverage=coverage,
        cumulative_return=cumulative,
        worst_day_return=worst_day,
    )


def worst_n_day_drawdown(returns: pd.Series, n: int) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    """Worst realized cumulative return over any rolling n-day window in the series,
    i.e. the worst real historical n-day drawdown for this return series.
    """
    cum = (1.0 + returns).cumprod()
    n_day_return = cum / cum.shift(n) - 1.0
    n_day_return = n_day_return.dropna()
    if n_day_return.empty:
        return float("nan"), None, None
    worst_end = n_day_return.idxmin()
    worst_value = float(n_day_return.loc[worst_end])
    worst_start = returns.index[returns.index.get_loc(worst_end) - n]
    return worst_value, worst_start, worst_end


def max_drawdown(returns: pd.Series) -> float:
    cum = (1.0 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1.0
    return float(drawdown.min())
