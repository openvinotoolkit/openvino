# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Configurations management
"""

# pylint: disable=fixme,broad-except

import os
import sys
import ast
import json
from pathlib import Path


if sys.version_info[:2] < (3, 9):
    raise Exception("Python version must be >= 3.9")


class ConfigException(Exception):
    """Base configuration exception"""


class Config:
    """Configuration wrapper"""

    _instance = None
    _properties = None

    default_cfg_path = Path(__file__).resolve().parent / "config.json"

    def __new__(cls, *_args, **_kwargs):
        if not Config._instance:
            Config._instance = super(Config, cls).__new__(cls)
        return Config._instance

    def __init__(self, file_path=None, cli_args=None):
        """
        :param file_path: Path to json configuration file
        :type file_path: String

        :param args: List of argparse arguments with patterns: 'name=value' or 'name'
        :type args: list
        """
        if Config._properties:
            return

        self._file_path = file_path or Config.default_cfg_path
        self._cli_args = cli_args or []

        self._json_cfg = {}
        self._args = {}

        self._load_cfg()
        self._parse_cli_args()

        Config._properties = {}
        for name, value in self._json_cfg.items():
            if hasattr(self, name):
                raise ConfigException(f"Duplicating prosperity: {name}")
            property_value = self._args.get(name) or os.getenv(name)
            if property_value:
                # Try to set prosperity_value as Python literal structures, e.g. DRY_RUN=False
                try:
                    property_value = ast.literal_eval(property_value)
                except Exception:
                    pass
                if not isinstance(property_value, type(value)):
                    raise ConfigException(f"Python type of {name} parameter must be {type(value)}")
            else:
                property_value = value
            Config._properties[name] = property_value

        self.set_proxy()

    def __getattr__(self, attr_name):
        if attr_name in self._properties:
            return self._properties.get(attr_name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr_name}'")

    def _load_cfg(self):
        """Load the json configuration file"""
        try:
            with open(self._file_path, encoding="utf-8") as conf:
                self._json_cfg = json.load(conf)
        except Exception as exc:
            raise ConfigException("Failed to load configuration from:", self._file_path) from exc

    def _parse_cli_args(self):
        """Parse argparse arguments with patterns: 'name=value' or 'name'"""
        for cli_arg in self._cli_args:
            arg = cli_arg.split("=")
            if arg[0] not in self._json_cfg:
                raise ConfigException(f"Unsupported argument: {arg}")
            self._args[arg[0]] = True if len(arg) == 1 else "=".join(arg[1:])

    @property
    def properties(self):
        """Get all properties as Dict"""
        return self._properties

    def set_proxy(self):
        """Set proxies"""
        for proxy_name, url in self._properties["PROXIES"].items():
            if url is not None:
                print(f"Set proxy: {proxy_name}={url}")
                os.environ[proxy_name] = url


def _test():
    """Test and debug"""
    print("Config.default_cfg_path:", Config.default_cfg_path)
    cfg = Config(cli_args=["DRY_RUN", 'PROXIES={"NO_PROXY": "localhost"}'])
    print("Config.properties:", cfg.properties)
    print("cfg.PROXIES:", cfg.PROXIES)


if __name__ == "__main__":
    _test()
