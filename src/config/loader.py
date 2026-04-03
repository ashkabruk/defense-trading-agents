from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.models.config import AgentsConfig, RiskConfig, Settings, SourcesConfig


@dataclass(slots=True)
class ConfigBundle:
    """Fully loaded and validated configuration set."""

    settings: Settings
    sources: SourcesConfig
    agents: AgentsConfig
    risk: RiskConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and return a mapping payload."""
    with path.open("r", encoding="utf-8") as file_obj:
        payload = yaml.safe_load(file_obj) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object at {path}, got {type(payload).__name__}")

    return payload


def load_config_bundle(config_dir: str | Path = "config") -> ConfigBundle:
    """Load all configuration files from a directory."""
    base_path = Path(config_dir)
    if not base_path.is_absolute():
        base_path = Path.cwd() / base_path

    settings = Settings.model_validate(_read_yaml(base_path / "settings.yaml"))
    sources = SourcesConfig.model_validate(_read_yaml(base_path / "sources.yaml"))
    agents = AgentsConfig.model_validate(_read_yaml(base_path / "agents.yaml"))
    risk = RiskConfig.model_validate(_read_yaml(base_path / "risk.yaml"))

    return ConfigBundle(settings=settings, sources=sources, agents=agents, risk=risk)
