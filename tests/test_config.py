import pytest
import yaml

from risk_dashboard.config import load_portfolio_config


def test_load_portfolio_config_roundtrip(tmp_path):
    cfg_path = tmp_path / "portfolio.yaml"
    cfg_path.write_text(
        yaml.dump(
            {
                "name": "test-portfolio",
                "assets": [
                    {"id": "A", "weight": 0.6, "role": "factor", "label": "Asset A"},
                    {"id": "B", "weight": 0.4, "role": "stock", "label": "Asset B"},
                ],
            }
        )
    )
    cfg = load_portfolio_config(cfg_path)
    assert cfg.name == "test-portfolio"
    assert cfg.weights == {"A": 0.6, "B": 0.4}
    assert cfg.ids == ["A", "B"]


def test_load_portfolio_config_rejects_bad_weight_sum(tmp_path):
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(
        yaml.dump(
            {
                "name": "bad-portfolio",
                "assets": [
                    {"id": "A", "weight": 0.6, "role": "factor", "label": "Asset A"},
                    {"id": "B", "weight": 0.6, "role": "stock", "label": "Asset B"},
                ],
            }
        )
    )
    with pytest.raises(ValueError):
        load_portfolio_config(cfg_path)
