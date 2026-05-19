import json
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str | None = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "config.json"
    else:
        config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if "threshold" not in data or "alpha" not in data:
        raise KeyError("Config must include 'threshold' and 'alpha'.")

    return data
