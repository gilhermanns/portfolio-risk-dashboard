from __future__ import annotations

import argparse
from pathlib import Path

from risk_dashboard import charts
from risk_dashboard.analysis import AnalysisResult, run_full_analysis
from risk_dashboard.config import load_portfolio_config

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Portfolio Risk Report</title>
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 980px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
  h1, h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: right; }}
  th {{ background: #f4f4f4; }}
  td:first-child, th:first-child {{ text-align: left; }}
  .chart {{ margin: 1.5rem 0; }}
  .note {{ background: #f7f2e0; border-left: 4px solid #b8933c; padding: 0.6rem 1rem; font-size: 0.9rem; }}
  code {{ background: #f4f4f4; padding: 1px 5px; }}
</style>
</head>
<body>
<h1>Portfolio Risk Report — {portfolio_name}</h1>
<p>Generated from real market data. Primary analysis window: {main_start} to {main_end}
({main_n} trading days). Extended factor-only analysis window: {ext_start} to {ext_end}.</p>

<h2>1. Portfolio composition</h2>
{weights_table}
<p>Herfindahl-Hirschman Index of weights: <b>{hhi:.4f}</b> (equal-weight-12 benchmark = {equal_hhi:.4f}).</p>

<h2>2. Value-at-Risk and Expected Shortfall (primary 12-asset portfolio, {main_start} to {main_end})</h2>
{var_table_main}
<div class="chart">{var_chart}</div>

<h2>3. Rolling risk</h2>
<div class="chart">{rolling_chart}</div>
<div class="chart">{cum_chart}</div>

<h2>4. Factor attribution</h2>
<p>Portfolio returns regressed on market (NASDAQ Composite), rates/bond-proxy (10Y Treasury), and
commodity (WTI oil) factors. R-squared = {r_squared:.3f}.</p>
{factor_table}

<h2>5. Correlation structure</h2>
<div class="chart">{corr_heatmap}</div>
<h3>Correlation regime: calm vs. crisis</h3>
<p>Average pairwise correlation, calm (2013-2014) vs. each crisis window. The 12-asset
column uses the full primary portfolio (available through the 2008 crisis only); the
4-factor column extends to 2020 and 2022 using just the factor-proxy assets.</p>
{regime_table}
<div class="chart">{regime_chart}</div>

<h2>6. Historical scenario stress tests</h2>
{stress_table}
<div class="chart">{stress_chart}</div>
<p class="note">2020 and 2022 scenarios use only the four factor-proxy assets (renormalized weights),
since free individual-stock data reachable in this environment does not extend past 2018-04-11.
See README "Limitations" for the full explanation.</p>

<h2>7. Worst historical drawdowns (primary portfolio)</h2>
{drawdown_table}

</body>
</html>
"""


def _weights_table(result: AnalysisResult) -> str:
    rows = "".join(
        f"<tr><td>{a.label}</td><td>{a.id}</td><td>{a.weight:.2%}</td><td>{a.role}</td></tr>"
        for a in result.config.assets
    )
    return f"<table><tr><th>Asset</th><th>ID</th><th>Weight</th><th>Role</th></tr>{rows}</table>"


def _df_table(df, pct_cols=None, fmt="{:.4f}") -> str:
    pct_cols = pct_cols or []
    header = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = ""
    for idx, row in df.iterrows():
        cells = "".join(
            f"<td>{row[c]:.2%}</td>" if c in pct_cols else f"<td>{fmt.format(row[c])}</td>" for c in df.columns
        )
        rows += f"<tr><td>{idx}</td>{cells}</tr>"
    return f"<table><tr><th></th>{header}</tr>{rows}</table>"


def _factor_table(result: AnalysisResult) -> str:
    fr = result.factor_result
    rows = "".join(
        f"<tr><td>{k}</td><td>{v:.4f}</td><td>{fr.variance_explained[k]:.2%}</td></tr>"
        for k, v in fr.betas.items()
    )
    return (
        f"<table><tr><th>Factor</th><th>Beta</th><th>% variance explained</th></tr>{rows}"
        f"<tr><td>Alpha (daily)</td><td>{fr.alpha:.6f}</td><td>-</td></tr>"
        f"<tr><td>Residual / idiosyncratic</td><td>-</td><td>{fr.residual_variance_share:.2%}</td></tr>"
        f"</table>"
    )


def _stress_table(result: AnalysisResult) -> str:
    rows = ""
    for r in result.scenario_results:
        rows += (
            f"<tr><td>{r.name}</td><td>{r.start}</td><td>{r.end}</td><td>{r.n_days}</td>"
            f"<td>{r.weight_coverage:.0%}</td><td>{r.cumulative_return:.2%}</td>"
            f"<td>{r.worst_day_return:.2%}</td></tr>"
        )
    return (
        "<table><tr><th>Scenario</th><th>Start</th><th>End</th><th>Days</th>"
        "<th>Weight coverage</th><th>Cumulative P&amp;L</th><th>Worst day</th></tr>"
        f"{rows}</table>"
    )


def _regime_table(result: AnalysisResult) -> str:
    """Combine the 12-asset (primary window) and 4-factor (extended window) correlation
    regime comparisons into one table, matching the README presentation. The 12-asset
    comparison only reaches the 2008 crisis (individual-stock history ends 2018-04-11),
    so later periods show a dash in that column rather than a misleading blank.
    """
    main = result.regime_main["avg_pairwise_corr"]
    extended = result.regime_extended["avg_pairwise_corr"]
    rows = ""
    for period in extended.index:
        main_cell = f"{main.loc[period]:.2%}" if period in main.index else "&mdash;"
        rows += f"<tr><td>{period}</td><td>{main_cell}</td><td>{extended.loc[period]:.2%}</td></tr>"
    return (
        "<table><tr><th>Period</th><th>Avg. pairwise correlation (12-asset)</th>"
        f"<th>Avg. pairwise correlation (4-factor)</th></tr>{rows}</table>"
    )


def _drawdown_table(result: AnalysisResult) -> str:
    rows = ""
    for n, (val, start, end) in result.worst_drawdowns.items():
        s = start.date() if start is not None else "-"
        e = end.date() if end is not None else "-"
        rows += f"<tr><td>{n}-day</td><td>{val:.2%}</td><td>{s}</td><td>{e}</td></tr>"
    return f"<table><tr><th>Window</th><th>Worst return</th><th>Start</th><th>End</th></tr>{rows}</table>"


def build_html_report(result: AnalysisResult) -> str:
    var_chart = charts.var_es_comparison_chart(result.var_table_main, "VaR comparison: historical vs. parametric vs. Monte Carlo")
    rolling_chart = charts.rolling_risk_chart(result.rolling_vol, result.rolling_var, "Rolling 63-day volatility and 250-day historical VaR")
    cum_chart = charts.cumulative_return_chart(result.main_portfolio_returns, "Cumulative growth of $1, primary portfolio")
    corr_heatmap = charts.correlation_heatmap(result.corr_matrix, "Asset correlation matrix (primary window)")
    regime_chart = charts.correlation_regime_chart(result.regime_extended, "Average pairwise correlation: calm vs. crisis windows")
    stress_chart = charts.stress_test_chart(result.scenario_results, "Historical scenario P&L on current weights")

    equal_hhi = 1.0 / len(result.config.ids)

    html = HTML_TEMPLATE.format(
        portfolio_name=result.config.name,
        main_start=result.main_window[0].date(),
        main_end=result.main_window[1].date(),
        main_n=len(result.main_returns),
        ext_start=result.extended_window[0].date(),
        ext_end=result.extended_window[1].date(),
        weights_table=_weights_table(result),
        hhi=result.hhi,
        equal_hhi=equal_hhi,
        var_table_main=_df_table(result.var_table_main, pct_cols=[c for c in result.var_table_main.columns]),
        var_chart=var_chart,
        rolling_chart=rolling_chart,
        cum_chart=cum_chart,
        r_squared=result.factor_result.r_squared,
        factor_table=_factor_table(result),
        corr_heatmap=corr_heatmap,
        regime_table=_regime_table(result),
        regime_chart=regime_chart,
        stress_table=_stress_table(result),
        stress_chart=stress_chart,
        drawdown_table=_drawdown_table(result),
    )
    return html


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a self-contained portfolio risk HTML report.")
    parser.add_argument("--config", required=True, help="Path to portfolio YAML config")
    parser.add_argument("--out", required=True, help="Path to write the HTML report")
    parser.add_argument("--offline", action="store_true", help="Skip live data fetch, use bundled cache only")
    parser.add_argument("--n-sims", type=int, default=20000, help="Monte Carlo simulation count")
    args = parser.parse_args()

    cfg = load_portfolio_config(args.config)
    result = run_full_analysis(cfg, use_live=not args.offline, n_sims=args.n_sims)
    html = build_html_report(result)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    print(f"Wrote report to {out_path}")


if __name__ == "__main__":
    main()
