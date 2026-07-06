import pandas as pd

from risk_dashboard.analysis import AnalysisResult
from risk_dashboard.config import AssetConfig, PortfolioConfig
from risk_dashboard.report import _regime_table


def _make_result(regime_main: pd.DataFrame, regime_extended: pd.DataFrame) -> AnalysisResult:
    """Build a minimal AnalysisResult stub with only the fields _regime_table reads,
    so this test doesn't need to run the full data pipeline.
    """
    cfg = PortfolioConfig(name="stub", assets=[AssetConfig(id="A", weight=1.0, role="factor", label="A")])
    return AnalysisResult(
        config=cfg,
        levels=pd.DataFrame(),
        returns=pd.DataFrame(),
        main_window=(None, None),
        main_returns=pd.DataFrame(),
        main_portfolio_returns=pd.Series(dtype=float),
        extended_window=(None, None),
        extended_returns=pd.DataFrame(),
        extended_weights={},
        extended_portfolio_returns=pd.Series(dtype=float),
        var_table_main=pd.DataFrame(),
        var_table_extended=pd.DataFrame(),
        rolling_vol=pd.Series(dtype=float),
        rolling_var=pd.Series(dtype=float),
        factor_result=None,
        corr_matrix=pd.DataFrame(),
        regime_main=regime_main,
        regime_extended=regime_extended,
        hhi=0.0,
    )


def test_regime_table_includes_both_12_asset_and_4_factor_columns():
    # regime_main only has data through the 2008 crisis (individual-stock history
    # stops in 2018); regime_extended covers all four periods. The HTML report must
    # surface both columns -- the 12-asset comparison is the one the README leads
    # with ("correlations go to 1 in a crisis"), so silently dropping it from the
    # generated report would make the report contradict its own README.
    regime_main = pd.DataFrame(
        {"avg_pairwise_corr": [0.264, 0.416]},
        index=pd.Index(["calm", "2008_financial_crisis"], name="period"),
    )
    regime_extended = pd.DataFrame(
        {"avg_pairwise_corr": [0.133, 0.060, 0.141, 0.194]},
        index=pd.Index(
            ["calm", "2008_financial_crisis", "2020_covid_crash", "2022_rate_shock"], name="period"
        ),
    )
    result = _make_result(regime_main, regime_extended)
    html = _regime_table(result)

    assert "26.40%" in html  # 12-asset calm figure must appear
    assert "41.60%" in html  # 12-asset 2008 figure must appear
    assert "13.30%" in html  # 4-factor calm figure
    # Periods with no 12-asset data (2020, 2022) must be shown as missing, not as 0%
    # or silently omitted from the table.
    assert html.count("&mdash;") == 2
    assert "2020_covid_crash" in html
    assert "2022_rate_shock" in html
