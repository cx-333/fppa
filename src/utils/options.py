# -*- encoding: utf-8 -*-
'''
YAML configuration loader with __base__ inheritance support.

@File    :   options.py
@Time    :   2026/07/04
@Author  :   XinChen (migrated)
'''

import os
import yaml
from copy import deepcopy


class DotDict(dict):
    """
    A dict subclass that supports attribute-style access (dot notation).
    Nested dicts are recursively converted to DotDict.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)
            elif isinstance(value, list):
                self[key] = [DotDict(item) if isinstance(item, dict) else item for item in value]

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute: '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute: '{key}'")


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge two dicts. override values take precedence."""
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def load_yaml_config(file_path: str) -> DotDict:
    """
    Read a YAML file and return a DotDict with __base__ inheritance resolved.

    If the YAML contains a __base__ key (a list of relative file paths),
    those base configs are loaded and deep-merged first (left to right),
    then the current file's other keys override the merged result.
    __base__ paths starting with '__base__/' are resolved relative to 'configs/';
    otherwise they are resolved relative to the current file's directory.
    """
    configs_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'configs')
    configs_dir = os.path.abspath(configs_dir)
    file_dir = os.path.dirname(os.path.abspath(file_path))

    with open(file_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    # Resolve __base__ inheritance
    base_paths = raw.pop('__base__', None)
    merged = {}

    if base_paths is not None:
        if not isinstance(base_paths, list):
            raise ValueError(f"__base__ must be a list, got {type(base_paths)} in {file_path}")

        for rel_path in base_paths:
            # Resolve relative to configs/ if starting with __base__, else relative to current file
            if rel_path.startswith('__base__/'):
                base_file = os.path.join(configs_dir, rel_path)
            else:
                base_file = os.path.join(file_dir, rel_path)
                base_file = os.path.normpath(base_file)

            if not os.path.exists(base_file):
                raise FileNotFoundError(f"Base config not found: {base_file}")
            base_config = load_yaml_config(base_file)
            merged = _deep_merge(merged, dict(base_config))

    # Current file's keys override base
    merged = _deep_merge(merged, raw)
    return DotDict(merged)
