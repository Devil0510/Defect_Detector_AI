import yaml
from pathlib import Path
from typing import Any, Dict

def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Loads and parses a YAML configuration file safely.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def save_yaml_config(config_data: Dict[str, Any], config_path: str | Path) -> None:
    """
    Saves a dictionary to a YAML configuration file.
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config_data, f, default_flow_style=False)
