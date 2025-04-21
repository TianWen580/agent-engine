import yaml
from typing import Any, Dict, Union

class ConfigNode:
    def __init__(self, data: Dict[str, Any]):
        self._data = {}
        self.raw = data
        for key, value in data.items():
            if isinstance(value, dict):
                self._data[key] = ConfigNode(value)
            else:
                self._data[key] = value

    def __getattr__(self, name: str) -> Union['ConfigNode', Any]:
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'ConfigNode' object has no attribute '{name}'")

    def __repr__(self) -> str:
        return f"<ConfigNode: {self._data.keys()}>"

def load_config(path: str) -> ConfigNode:
    with open(path, 'r') as f:
        raw_data = yaml.safe_load(f)
    return ConfigNode(raw_data)
