from __future__ import annotations
import argparse as argparse
from openvino.tools.ovc.error import Error
import os as os
import pathlib
from pathlib import Path
__all__ = ['Error', 'Path', 'any_extensions_used', 'argparse', 'default_path', 'get_transformations_config_path', 'legacy_transformations_config_used', 'os', 'tensorflow_custom_operations_config_update_used']
def any_extensions_used(argv: argparse.Namespace):
    ...
def default_path():
    ...
def get_transformations_config_path(argv: argparse.Namespace) -> pathlib.Path:
    ...
def legacy_transformations_config_used(argv: argparse.Namespace):
    ...
def tensorflow_custom_operations_config_update_used(argv: argparse.Namespace):
    ...
