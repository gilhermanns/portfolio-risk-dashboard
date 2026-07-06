from __future__ import annotations

import numpy as np
import pandas as pd


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()


def average_pairwise_correlation(returns: pd.DataFrame) -> float:
    corr = returns.corr().to_numpy()
    n = corr.shape[0]
    if n < 2:
        return float("nan")
    off_diag_sum = corr.sum() - np.trace(corr)
    count = n * (n - 1)
    return float(off_diag_sum / count)


def correlation_regime_comparison(
    returns: pd.DataFrame, calm_window: tuple[str, str], stress_windows: dict[str, tuple[str, str]]
) -> pd.DataFrame:
    """Average pairwise correlation in a calm period vs one or more stress periods,
    illustrating the "correlations go to 1 in a crisis" effect.
    """
    rows = []
    calm_slice = returns.loc[calm_window[0] : calm_window[1]]
    rows.append({"period": "calm", "start": calm_window[0], "end": calm_window[1],
                 "avg_pairwise_corr": average_pairwise_correlation(calm_slice), "n_obs": len(calm_slice)})
    for name, (start, end) in stress_windows.items():
        stress_slice = returns.loc[start:end]
        rows.append({"period": name, "start": start, "end": end,
                     "avg_pairwise_corr": average_pairwise_correlation(stress_slice), "n_obs": len(stress_slice)})
    return pd.DataFrame(rows).set_index("period")
