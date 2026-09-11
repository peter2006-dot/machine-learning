"""Project configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a TOML configuration file and validate shared project settings."""
    config_path = Path(path)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    required_sections = {"project", "data", "features", "model", "logging"}
    missing = required_sections.difference(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")

    train = float(config["data"]["train_ratio"])
    validation = float(config["data"]["validation_ratio"])
    test = float(config["data"]["test_ratio"])
    if abs(train + validation + test - 1.0) > 1e-8:
        raise ValueError("train_ratio, validation_ratio and test_ratio must sum to 1")

    if config["data"]["label_transform"] != "log10":
        raise ValueError("The initial pipeline supports only the log10 label transform")

    return config
