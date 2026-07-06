from __future__ import annotations

import numpy as np
import pandas as pd

from risk_dashboard.var import historical_var


def rolling_volatility(returns: pd.Series, window: int = 63, annualize: bool = True) -> pd.Series:
    vol = returns.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol.rename("rolling_vol")


def rolling_historical_var(returns: pd.Series, window: int = 250, alpha: float = 0.95) -> pd.Series:
    return returns.rolling(window).apply(lambda w: historical_var(w, alpha), raw=True).rename("rolling_var")
