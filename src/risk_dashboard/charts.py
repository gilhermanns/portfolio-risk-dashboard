from __future__ import annotations

import io
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

PALETTE = ["#3b6ea5", "#c96f3d", "#5a9367", "#a15b8f", "#b8933c", "#5f6a72"]


def fig_to_svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    svg = buf.getvalue()
    match = re.search(r"<svg[\s\S]*</svg>", svg)
    return match.group(0) if match else svg


def cumulative_return_chart(returns: pd.Series, title: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    cum = (1.0 + returns).cumprod()
    ax.plot(cum.index, cum.to_numpy(), color=PALETTE[0], linewidth=1.2)
    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig_to_svg(fig)


def rolling_risk_chart(rolling_vol: pd.Series, rolling_var: pd.Series, title: str) -> str:
    fig, ax1 = plt.subplots(figsize=(8, 3.5))
    ax1.plot(rolling_vol.index, rolling_vol.to_numpy(), color=PALETTE[0], linewidth=1.0, label="Annualized rolling volatility")
    ax1.set_ylabel("Annualized volatility")
    ax2 = ax1.twinx()
    ax2.plot(rolling_var.index, rolling_var.to_numpy(), color=PALETTE[1], linewidth=1.0, label="1-day 95% historical VaR")
    ax2.set_ylabel("Daily VaR (95%)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8, frameon=False)
    ax1.set_title(title)
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    return fig_to_svg(fig)


def var_es_comparison_chart(var_table: pd.DataFrame, title: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    methods = ["historical", "parametric_normal", "parametric_t", "monte_carlo"]
    method_labels = ["Historical", "Parametric (Normal)", "Parametric (Student-t)", "Monte Carlo"]
    confidences = var_table.index.tolist()
    x = np.arange(len(confidences))
    width = 0.19
    for i, (method, label) in enumerate(zip(methods, method_labels)):
        col = f"{method}_VaR"
        if col not in var_table.columns:
            continue
        vals = var_table[col].to_numpy() * 100
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=label, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(c*100)}%" for c in confidences])
    ax.set_ylabel("1-day VaR (% of portfolio)")
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    return fig_to_svg(fig)


def correlation_heatmap(corr: pd.DataFrame, title: str) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=7)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6, color="black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
    ax.set_title(title)
    fig.tight_layout()
    return fig_to_svg(fig)


def correlation_regime_chart(regime_df, title: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = [PALETTE[0]] + [PALETTE[1]] * (len(regime_df) - 1)
    ax.bar(regime_df.index, regime_df["avg_pairwise_corr"], color=colors)
    ax.set_ylabel("Average pairwise correlation")
    ax.set_title(title)
    ax.grid(alpha=0.25, axis="y")
    for i, v in enumerate(regime_df["avg_pairwise_corr"]):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    return fig_to_svg(fig)


def stress_test_chart(scenario_results, title: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    names = [r.name for r in scenario_results]
    values = [r.cumulative_return * 100 for r in scenario_results]
    colors = [PALETTE[2] if v >= 0 else PALETTE[1] for v in values]
    ax.bar(names, values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Cumulative portfolio return (%)")
    ax.set_title(title)
    ax.grid(alpha=0.25, axis="y")
    for i, v in enumerate(values):
        ax.text(i, v + (1 if v >= 0 else -2), f"{v:.1f}%", ha="center", fontsize=8)
    fig.tight_layout()
    return fig_to_svg(fig)
