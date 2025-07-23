# type: ignore
from __future__ import annotations
from builtins import module as ModuleType
from collections.abc import Callable
from functools import wraps
from pathlib import Path
import collections.abc
import importlib as importlib
import os as os
import sys as sys
import typing
__all__ = ['Callable', 'LazyLoader', 'ModuleType', 'Path', 'classproperty', 'deprecated', 'deprecatedclassproperty', 'get_cmake_path', 'importlib', 'os', 'sys', 'wraps']
class LazyLoader:
    """
    A class to lazily load a module, importing it only when an attribute is accessed.
    """
    def __getattr__(self, item: str) -> typing.Any:
        ...
    def __init__(self, module_name: str):
        ...
    def __repr__(self) -> str:
        ...
    def _load_module(self) -> None:
        ...
class _ClassPropertyDescriptor:
    def __get__(self, obj: typing.Any, cls: typing.Any = None) -> typing.Any:
        ...
    def __init__(self, fget: collections.abc.Callable):
        ...
def _add_openvino_libs_to_search_path() -> None:
    """
    Add OpenVINO libraries to the DLL search path on Windows.
    """
def classproperty(func: typing.Any) -> _ClassPropertyDescriptor:
    ...
def deprecated(name: typing.Any = None, version: str = '', message: str = '', stacklevel: int = 2) -> collections.abc.Callable[..., typing.Any]:
    """
    Prints deprecation warning "{function_name} is deprecated and will be removed in version {version}. {message}" and runs the function.
    
        :param version: The version in which the code will be removed.
        :param message: A message explaining why the function is deprecated and/or what to use instead.
        
    """
def deprecatedclassproperty(name: typing.Any = None, version: str = '', message: str = '', stacklevel: int = 2) -> collections.abc.Callable[[typing.Any], _ClassPropertyDescriptor]:
    ...
def get_cmake_path() -> str:
    """
    Searches for the directory containing CMake files within the package install directory.
    
        :return: The path to the directory containing CMake files, if found. Otherwise, returns empty string.
        :rtype: str
        
    """
