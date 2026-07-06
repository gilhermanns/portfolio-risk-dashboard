"""Data acquisition: live fetch from FRED / a public historical-price dataset, with a
bundled cached-CSV fallback so the dashboard still runs offline or if a source is
rate-limited or otherwise unreachable at runtime.

Ticker universe and why: this project targets stooq.com and Yahoo Finance for
SPY/QQQ/TLT/GLD/individual-stock OHLCV, but both were unreachable from the build
environment (stooq requires a client-side JS proof-of-work challenge; Yahoo's
endpoints were persistently rate-limited at the network level). All data below is
instead sourced from the Federal Reserve Economic Data (FRED) API and a public
domain historical-stock-price CSV, both fetchable without an API key. See the
README "Limitations" section for the full explanation and the substitutions used
(e.g. NASDAQ Composite as the broad-equity factor in place of SPY).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE_DIR = Path(__file__).parent / "cache_data"

FRED_SERIES = {
    "MKT_NASDAQCOM": "NASDAQCOM",
    "GROWTH_NASDAQ100": "NASDAQ100",
    "BOND_DGS10": "DGS10",
    "OIL_WTI": "DCOILWTICO",
}
FRED_CACHE_FILE = {
    "MKT_NASDAQCOM": "nasdaqcom.csv",
    "GROWTH_NASDAQ100": "nasdaq100.csv",
    "BOND_DGS10": "dgs10.csv",
    "OIL_WTI": "dcoilwtico.csv",
}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

STOCK_PRICES_URL = (
    "https://raw.githubusercontent.com/robertmartin8/PyPortfolioOpt/master/"
    "tests/resources/stock_prices.csv"
)
STOCK_TICKERS = ["AAPL", "XOM", "JPM", "PFE", "WMT", "GE", "BAC", "T"]

BOND_DURATION_YEARS = 7.5  # approximate modified duration of a constant 10Y CMT position


def _fetch_text(url: str, timeout: float = 15.0) -> str | None:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        text = resp.text
        if text.lstrip().startswith("<!DOCTYPE") or text.lstrip().startswith("<html"):
            return None  # anti-bot / error page, not real data
        return text
    except requests.RequestException:
        return None


def _read_fred_csv(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text), na_values=["."])
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["value"].astype(float)


def load_fred_series(asset_id: str, use_live: bool = True) -> pd.Series:
    """Load one FRED series as a level (index/yield) series, live-first with cache fallback."""
    series_id = FRED_SERIES[asset_id]
    text = _fetch_text(FRED_URL.format(series_id=series_id)) if use_live else None
    source = "live"
    if text is None:
        cache_path = CACHE_DIR / FRED_CACHE_FILE[asset_id]
        text = cache_path.read_text()
        source = "cache"
    series = _read_fred_csv(text)
    series.name = asset_id
    series.attrs["source"] = source
    return series


def load_stock_prices(use_live: bool = True) -> pd.DataFrame:
    """Load individual-name adjusted close prices, live-first with cache fallback."""
    text = _fetch_text(STOCK_PRICES_URL) if use_live else None
    source = "live"
    if text is None:
        text = (CACHE_DIR / "stock_prices.csv").read_text()
        source = "cache"
    df = pd.read_csv(io.StringIO(text))
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.attrs["source"] = source
    return df[STOCK_TICKERS]


def bond_return_from_yield(yield_pct: pd.Series, duration: float = BOND_DURATION_YEARS) -> pd.Series:
    """Approximate daily total-return series for a constant-maturity Treasury position
    from its yield level, combining daily coupon carry with a duration-based price
    response to yield changes: r_t = y_{t-1}/100/252 - duration * (y_t - y_{t-1})/100.

    This is a standard yield-to-return approximation used when a literal total-return
    series (e.g. TLT) is unavailable; it assumes constant duration and ignores convexity.
    """
    y = yield_pct.astype(float) / 100.0
    carry = y.shift(1) / 252.0
    price_effect = -duration * y.diff()
    return (carry + price_effect).rename("BOND_DGS10")


def build_price_levels(use_live: bool = True) -> pd.DataFrame:
    """Assemble a single DataFrame of asset 'levels' (index points, or 100-based level
    for the synthetic bond series) aligned on a common daily calendar, inner-joined so
    every column has a real observation on every retained date.
    """
    factor_series = {aid: load_fred_series(aid, use_live=use_live) for aid in FRED_SERIES}
    stocks = load_stock_prices(use_live=use_live)

    bond_ret = bond_return_from_yield(factor_series["BOND_DGS10"])
    bond_level = 100.0 * (1.0 + bond_ret.fillna(0)).cumprod()
    bond_level.iloc[0] = np.nan  # first observation has no prior yield to diff against

    frame = pd.DataFrame(
        {
            "MKT_NASDAQCOM": factor_series["MKT_NASDAQCOM"],
            "GROWTH_NASDAQ100": factor_series["GROWTH_NASDAQ100"],
            "BOND_DGS10": bond_level,
            "OIL_WTI": factor_series["OIL_WTI"],
        }
    )
    frame = frame.join(stocks, how="outer")
    frame = frame.sort_index()
    return _fill_calendar_gaps(frame)


def _fill_calendar_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill sporadic single-source calendar mismatches (e.g. a day the bond
    market is closed but equity indices trade, or vice versa) within each column's
    own observed date range, leaving genuine before-inception / after-discontinuation
    gaps untouched so `common_window` still reflects true data availability.
    """
    filled = frame.copy()
    for col in filled.columns:
        first, last = filled[col].first_valid_index(), filled[col].last_valid_index()
        if first is None:
            continue
        filled.loc[first:last, col] = filled.loc[first:last, col].ffill()
    return filled


def common_window(levels: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Restrict to the inner-joined date range where every requested column is populated."""
    cols = columns or list(levels.columns)
    sub = levels[cols].dropna(how="any")
    return sub
