# Portfolio Risk Dashboard

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive tool for **quantifying and visualizing multi-asset portfolio risk**, crucial for **Wealth Management, Private Banking, and Institutional Risk Management**. This dashboard moves beyond simplistic risk metrics to provide a nuanced understanding of portfolio vulnerabilities, especially during periods of market stress.

## Business Relevance

For financial professionals in **Wealth Management, Private Banking, and Institutional Risk Management**, accurately assessing and communicating portfolio risk is paramount. This dashboard offers capabilities vital for:

-   **Client Advisory**: Providing clients with transparent and comprehensive insights into their portfolio's risk profile, fostering trust and informed decision-making.
-   **Regulatory Compliance**: Meeting stringent risk reporting requirements by employing multiple methodologies for Value-at-Risk (VaR) and Expected Shortfall (ES).
-   **Stress Testing**: Proactively evaluating portfolio resilience against historical and hypothetical market shocks, a cornerstone of robust risk management.
-   **Strategic Asset Allocation**: Informing allocation decisions by understanding factor exposures, correlation dynamics, and concentration risks.
-   **Product Development**: Enhancing investment product design with rigorous risk analytics and clear visualization of potential downside scenarios.

This project demonstrates a practical approach to institutional-grade risk analysis, moving beyond theoretical assumptions to real-world market behavior.

## Motivation (Applied Perspective)

Traditional risk models often rely on assumptions (e.g., normal distribution of returns) that break down precisely when risk management is most critical – during market crises. This project addresses the limitations of point-estimate risk models by:

-   **Comparing Multiple Methodologies**: Quantifying Value-at-Risk (VaR) and Expected Shortfall (ES) using historical simulation, parametric (variance-covariance), and Monte Carlo approaches. This highlights how different models can diverge, particularly in tail events.
-   **Analyzing Crisis Regimes**: Utilizing real historical events (2008 financial crisis, 2020 COVID crash, 2022 rate-shock selloff) as test cases to demonstrate where each risk methodology succeeds or fails.
-   **Understanding Risk Dynamics**: Tracking rolling volatility and correlation regimes to show how portfolio risk evolves from calm periods to stressed environments.

This dashboard provides a robust framework for understanding and communicating complex portfolio risks, enabling more informed decision-making for both advisors and clients.

## Methodology

-   **Portfolio**: A diversified 12-asset portfolio (`portfolio.yaml`) comprising broad market factors (equity, growth equity, bond proxy, commodity) and individual stocks across various sectors (technology, energy, financials, healthcare, consumer staples, industrials, telecom).
-   **VaR / Expected Shortfall**: Calculated at 95% and 99% confidence levels using three distinct methods:
    -   *Historical Simulation*: Non-parametric, directly from empirical returns.
    -   *Parametric (Variance-Covariance)*: Assumes Normal and Student-t distributions, highlighting the impact of fat tails.
    -   *Monte Carlo*: Simulates daily portfolio returns based on a multivariate Normal fit, providing a robust statistical estimate.
-   **Rolling Risk**: Annualized volatility and historical VaR tracked over a 250-day rolling window to capture dynamic risk changes.
-   **Factor Attribution**: OLS regression of portfolio returns against key factors (equity market, rates/bond proxy, commodity) to identify primary risk drivers and their contribution to portfolio variance.
-   **Correlation Regimes**: Compares average pairwise correlations during calm periods versus crisis windows, directly testing the 
"correlations go to 1 in a crisis" hypothesis.
-   **Concentration**: Herfindahl-Hirschman Index (HHI) to measure portfolio concentration.
-   **Historical Stress Tests**: Replays actual realized return sequences from major historical events (2008, 2020, 2022) against the current portfolio weights to assess resilience.

## Sample Output & Insights

This project generates actionable insights and visualizations, essential for portfolio analysis and client communication:

-   **VaR / Expected Shortfall Comparison**: Illustrates the divergence of different risk models, particularly the underestimation of tail risk by parametric models during crises.
-   **Factor Attribution**: Quantifies the portfolio's sensitivity to key market factors and identifies the primary drivers of its variance.
-   **Correlation Regime Analysis**: Demonstrates how asset correlations shift dramatically during periods of market stress, impacting diversification benefits.
-   **Historical Stress Test Results**: Provides concrete figures for portfolio performance during major market downturns, offering a realistic view of potential losses.
-   **Worst Realized Drawdowns**: Highlights the most severe historical drawdowns over various time horizons.

*(Example charts and tables would be embedded here, showcasing VaR/ES comparisons, rolling risk metrics, factor attributions, and stress test results.)*

## Project Structure

```text
/portfolio-risk-dashboard
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── portfolio.yaml          # Portfolio definition (weights and assets)
├── src/
│   ├── risk_dashboard/     # Core risk analysis logic
│   │   ├── data.py         # Data fetching and processing
│   │   ├── metrics.py      # VaR, ES, and other risk metrics calculations
│   │   ├── report.py       # HTML report generation
│   │   └── cache_data/     # Bundled cached data for offline use
├── tests/                  # Pytest suite
└── reports/                # Generated HTML reports and images
```

## Getting Started

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/gilhermanns/portfolio-risk-dashboard.git
   cd portfolio-risk-dashboard
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Generate the risk report:
   ```bash
   python -m risk_dashboard.report --config portfolio.yaml --out reports/risk_report.html --n-sims 50000
   ```
   (Use `--offline` to force bundled cached data and avoid network calls.)

## License & Disclaimer

This project is licensed under the MIT License. It is intended for educational and research purposes in quantitative finance. The models and results presented are for illustrative purposes and do not constitute financial advice or guarantee real-world performance. Always exercise professional judgment and conduct thorough due diligence when managing financial portfolios.
