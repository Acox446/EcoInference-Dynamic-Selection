"""
Configuration loader for EcoInference.
Loads YAML configs from the configs/ directory.
"""
import os
import yaml
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"


def load_config(name: str) -> dict:
    """
    Load a YAML config file by name.
    
    Args:
        name: Config name without extension (e.g., "models", "experiments")
    
    Returns:
        Dictionary with config values
    """
    config_path = _CONFIGS_DIR / f"{name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_models_config() -> dict:
    """Load models configuration."""
    return load_config("models")


def get_experiments_config() -> dict:
    """Load experiments configuration."""
    return load_config("experiments")


def get_energy_costs() -> dict:
    """Get energy costs dict directly."""
    return load_config("models")["energy_costs"]


def get_active_models() -> list:
    """Get list of active model names."""
    return load_config("experiments")["active_models"]


def get_cascading_thresholds(preset: str = None) -> list:
    """
    Get cascading thresholds.
    
    Args:
        preset: Optional preset name ("aggressive", "balanced", "conservative")
                If None, returns default thresholds.
    """
    config = load_config("experiments")["cascading"]
    
    if preset:
        return config["presets"][preset]
    return config["thresholds"]


def get_paths() -> dict:
    """Get paths configuration."""
    return load_config("experiments")["paths"]
