import yaml
from typing import Any, Dict, Union, List

class ConfigNode:
    def __init__(self, data: Union[Dict[str, Any], List[Any]]):
        self._data = {}
        self.raw = data

        if isinstance(data, dict):  # 字典
            for key, value in data.items():
                self._data[key] = self._process_value(value)
        elif isinstance(data, list):  # 列表
            self._data = [self._process_value(item) for item in data]
        else:  # 简单类型
            self._data = data

    def _process_value(self, value: Any) -> Union['ConfigNode', Any]:
        """
        递归地处理嵌套结构。
        """
        if isinstance(value, dict):  # 嵌套字典
            return ConfigNode(value)
        elif isinstance(value, list):  # 嵌套列表
            return [self._process_value(item) for item in value]
        else:  # 简单类型（如字符串、数字、布尔值、None）
            return value

    def __getattr__(self, name: str) -> Union['ConfigNode', Any]:
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'ConfigNode' object has no attribute '{name}'")

    def __getitem__(self, key: Union[str, int]) -> Union['ConfigNode', Any]:
        """
        支持通过索引访问列表或字典。
        """
        if isinstance(self._data, dict) and key in self._data:
            return self._data[key]
        elif isinstance(self._data, list) and isinstance(key, int):
            return self._data[key]
        raise KeyError(f"Key or index '{key}' not found in ConfigNode")

    def __repr__(self) -> str:
        if isinstance(self._data, dict):
            return f"<ConfigNode: {list(self._data.keys())}>"
        elif isinstance(self._data, list):
            return f"<ConfigNode: List of length {len(self._data)}>"
        else:
            return f"<ConfigNode: {self._data}>"

def load_config(path: str) -> ConfigNode:
    with open(path, 'r') as f:
        raw_data = yaml.safe_load(f)
    return ConfigNode(raw_data)
