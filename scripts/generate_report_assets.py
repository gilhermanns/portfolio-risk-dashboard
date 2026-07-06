import json

from risk_dashboard import charts
from risk_dashboard.analysis import run_full_analysis
from risk_dashboard.config import load_portfolio_config

cfg = load_portfolio_config("portfolio.yaml")
r = run_full_analysis(cfg, use_live=False, n_sims=50000)

svg_cum = charts.cumulative_return_chart(r.main_portfolio_returns, "Cumulative growth of $1 - primary 12-asset portfolio (1989-2018)")
svg_var = charts.var_es_comparison_chart(r.var_table_main, "VaR comparison: historical vs. parametric vs. Monte Carlo (primary portfolio)")
svg_roll = charts.rolling_risk_chart(r.rolling_vol, r.rolling_var, "Rolling 63-day volatility and 250-day historical VaR")
svg_regime = charts.correlation_regime_chart(r.regime_extended, "Average pairwise correlation: calm vs. crisis (factor-only, 1986-2026)")
svg_heatmap = charts.correlation_heatmap(r.corr_matrix, "Asset correlation matrix (primary 12-asset window)")
svg_stress = charts.stress_test_chart(r.scenario_results, "Historical scenario P&L on current portfolio weights")

with open("docs/img/cumulative_return.svg", "w") as f:
    f.write(svg_cum)
with open("docs/img/var_comparison.svg", "w") as f:
    f.write(svg_var)
with open("docs/img/rolling_risk.svg", "w") as f:
    f.write(svg_roll)
with open("docs/img/correlation_regime.svg", "w") as f:
    f.write(svg_regime)
with open("docs/img/correlation_heatmap.svg", "w") as f:
    f.write(svg_heatmap)
with open("docs/img/stress_test.svg", "w") as f:
    f.write(svg_stress)

results = {
    "main_window": [str(r.main_window[0].date()), str(r.main_window[1].date())],
    "extended_window": [str(r.extended_window[0].date()), str(r.extended_window[1].date())],
    "var_table_main": r.var_table_main.round(6).to_dict(),
    "var_table_extended": r.var_table_extended.round(6).to_dict(),
    "hhi": r.hhi,
    "equal_weight_hhi": 1 / len(cfg.ids),
    "factor_alpha_daily": r.factor_result.alpha,
    "factor_betas": r.factor_result.betas,
    "factor_r_squared": r.factor_result.r_squared,
    "factor_variance_explained": r.factor_result.variance_explained,
    "factor_residual_share": r.factor_result.residual_variance_share,
    "regime_main": r.regime_main.reset_index().to_dict(orient="records"),
    "regime_extended": r.regime_extended.reset_index().to_dict(orient="records"),
    "scenarios": [
        {
            "name": s.name,
            "start": s.start,
            "end": s.end,
            "n_days": s.n_days,
            "assets_used": s.assets_used,
            "weight_coverage": s.weight_coverage,
            "cumulative_return": s.cumulative_return,
            "worst_day_return": s.worst_day_return,
        }
        for s in r.scenario_results
    ],
    "worst_drawdowns": {
        str(n): [v, str(st.date()) if st is not None else None, str(en.date()) if en is not None else None]
        for n, (v, st, en) in r.worst_drawdowns.items()
    },
    "ann_return_main": float(r.main_portfolio_returns.mean() * 252),
    "ann_vol_main": float(r.main_portfolio_returns.std() * (252**0.5)),
}

with open("scratch_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(json.dumps(results, indent=2, default=str))
