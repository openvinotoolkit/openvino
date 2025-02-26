from __future__ import annotations
from builtins import module as ModuleType
from functools import wraps
import importlib as importlib
import os as os
from pathlib import Path
import sys as sys
__all__ = ['ModuleType', 'Path', 'classproperty', 'deprecated', 'deprecatedclassproperty', 'get_cmake_path', 'importlib', 'lazy_import', 'os', 'sys', 'wraps']
class _ClassPropertyDescriptor:
    def __get__(self, obj: typing.Any, cls: typing.Any = None) -> typing.Any:
        ...
    def __init__(self, fget: typing.Callable):
        ...
def _add_openvino_libs_to_search_path() -> None:
    """
    Add OpenVINO libraries to the DLL search path on Windows.
    """
def classproperty(func: typing.Any) -> _ClassPropertyDescriptor:
    ...
def deprecated(name: typing.Any = None, version: str = '', message: str = '', stacklevel: int = 2) -> typing.Callable[..., typing.Any]:
    """
    Prints deprecation warning "{function_name} is deprecated and will be removed in version {version}. {message}" and runs the function.
    
        :param version: The version in which the code will be removed.
        :param message: A message explaining why the function is deprecated and/or what to use instead.
        
    """
def deprecatedclassproperty(name: typing.Any = None, version: str = '', message: str = '', stacklevel: int = 2) -> typing.Callable[[typing.Any], openvino.package_utils._ClassPropertyDescriptor]:
    ...
def get_cmake_path() -> str:
    """
    Searches for the directory containing CMake files within the package install directory.
    
        :return: The path to the directory containing CMake files, if found. Otherwise, returns empty string.
        :rtype: str
        
    """
def lazy_import(module_name: str) -> module:
    ...
