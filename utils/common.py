"""Configuration helpers used by the training entry point."""

from typing import Mapping, Any
import importlib


def get_obj_from_str(string: str, reload: bool = False) -> Any:
    module, cls = string.rsplit(".", 1)
    if reload:
        importlib.reload(importlib.import_module(module))
    return getattr(importlib.import_module(module), cls)


def instantiate_from_config(config: Mapping[str, Any]) -> Any:
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", {}))
