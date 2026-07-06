from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from risk_dashboard import correlation, data, factors, portfolio, scenarios, stress, var
from risk_dashboard.config import PortfolioConfig
from risk_dashboard.rolling import rolling_historical_var, rolling_volatility

FACTOR_COLUMNS = ["MKT_NASDAQCOM", "BOND_DGS10", "OIL_WTI"]


@dataclass
class AnalysisResult:
    config: PortfolioConfig
    levels: pd.DataFrame
    returns: pd.DataFrame
    main_window: tuple
    main_returns: pd.DataFrame
    main_portfolio_returns: pd.Series
    extended_window: tuple
    extended_returns: pd.DataFrame
    extended_weights: dict
    extended_portfolio_returns: pd.Series
    var_table_main: pd.DataFrame
    var_table_extended: pd.DataFrame
    rolling_vol: pd.Series
    rolling_var: pd.Series
    factor_result: factors.FactorRegressionResult
    corr_matrix: pd.DataFrame
    regime_main: pd.DataFrame
    regime_extended: pd.DataFrame
    hhi: float
    scenario_results: list = field(default_factory=list)
    worst_drawdowns: dict = field(default_factory=dict)


def run_full_analysis(cfg: PortfolioConfig, use_live: bool = True, n_sims: int = 20000) -> AnalysisResult:
    levels = data.build_price_levels(use_live=use_live)
    returns = portfolio.simple_returns(levels)

    weights = cfg.weights
    main_common = data.common_window(levels, cfg.ids)
    main_returns = portfolio.simple_returns(main_common)
    main_port_returns = portfolio.portfolio_returns(main_returns, weights)
    main_window = (main_common.index.min(), main_common.index.max())

    factor_cols = [a for a in cfg.ids if a in data.FRED_SERIES]
    extended_common = data.common_window(levels, factor_cols)
    extended_returns = portfolio.simple_returns(extended_common)
    extended_weights = portfolio.renormalized_weights(weights, factor_cols)
    extended_port_returns = portfolio.portfolio_returns(extended_returns, extended_weights)
    extended_window = (extended_common.index.min(), extended_common.index.max())

    var_table_main = var.var_es_table(
        main_port_returns, asset_returns=main_returns, weights=weights, n_sims=n_sims
    )
    var_table_extended = var.var_es_table(
        extended_port_returns, asset_returns=extended_returns, weights=extended_weights, n_sims=n_sims
    )

    roll_vol = rolling_volatility(main_port_returns, window=63)
    roll_var = rolling_historical_var(main_port_returns, window=250, alpha=0.95)

    factor_returns = main_returns[[c for c in FACTOR_COLUMNS if c in main_returns.columns]]
    factor_result = factors.factor_regression(main_port_returns, factor_returns)

    corr_matrix = correlation.correlation_matrix(main_returns)

    regime_main = correlation.correlation_regime_comparison(
        main_returns, scenarios.CALM_WINDOW, {"2008_financial_crisis": scenarios.STRESS_WINDOWS["2008_financial_crisis"]}
    )
    regime_extended = correlation.correlation_regime_comparison(
        extended_returns, scenarios.CALM_WINDOW, scenarios.STRESS_WINDOWS
    )

    hhi = portfolio.herfindahl_index(weights)

    scenario_results = []
    for name, (start, end) in scenarios.STRESS_WINDOWS.items():
        scenario_results.append(stress.apply_historical_scenario(returns, weights, name, start, end))

    worst_drawdowns = {}
    for n in (1, 5, 10, 20):
        val, wstart, wend = stress.worst_n_day_drawdown(main_port_returns, n)
        worst_drawdowns[n] = (val, wstart, wend)

    return AnalysisResult(
        config=cfg,
        levels=levels,
        returns=returns,
        main_window=main_window,
        main_returns=main_returns,
        main_portfolio_returns=main_port_returns,
        extended_window=extended_window,
        extended_returns=extended_returns,
        extended_weights=extended_weights,
        extended_portfolio_returns=extended_port_returns,
        var_table_main=var_table_main,
        var_table_extended=var_table_extended,
        rolling_vol=roll_vol,
        rolling_var=roll_var,
        factor_result=factor_result,
        corr_matrix=corr_matrix,
        regime_main=regime_main,
        regime_extended=regime_extended,
        hhi=hhi,
        scenario_results=scenario_results,
        worst_drawdowns=worst_drawdowns,
    )
