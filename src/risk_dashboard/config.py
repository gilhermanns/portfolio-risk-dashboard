from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class AssetConfig:
    id: str
    weight: float
    role: str
    label: str


@dataclass
class PortfolioConfig:
    name: str
    assets: list[AssetConfig]

    @property
    def weights(self) -> dict[str, float]:
        return {a.id: a.weight for a in self.assets}

    @property
    def labels(self) -> dict[str, str]:
        return {a.id: a.label for a in self.assets}

    @property
    def ids(self) -> list[str]:
        return [a.id for a in self.assets]


def load_portfolio_config(path: str | Path) -> PortfolioConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    assets = [
        AssetConfig(id=a["id"], weight=float(a["weight"]), role=a.get("role", "asset"), label=a.get("label", a["id"]))
        for a in raw["assets"]
    ]
    total = sum(a.weight for a in assets)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Portfolio weights must sum to 1.0, got {total:.6f}")
    return PortfolioConfig(name=raw.get("name", "portfolio"), assets=assets)
