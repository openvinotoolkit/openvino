# type: ignore
from builtins import builtin_function_or_method as BuiltinFunctionType
from builtins import module as ModuleType
from __future__ import annotations
from typing import Any
import sys as sys
import typing
__all__ = ['Any', 'BuiltinFunctionType', 'ModuleType', 'Property', 'sys']
class Property(str):
    """
    This class allows to make a string object callable. Call returns underlying string's data.
    """
    @classmethod
    def __new__(cls, prop: typing.Callable[..., typing.Any]):
        ...
    def __call__(self, *args: typing.Any) -> typing.Callable[..., typing.Any]:
        ...
def __append_property_to_module(func: typing.Callable[..., typing.Any], target_module_name: str) -> None:
    """
    Modifies the target module's __getattr__ method to expose a python property wrapper by the function's name.
    
        :param func: the function which will be transformed to behave as python property.
        :param target_module_name: the name of the module to which properties are added.
        
    """
def __make_properties(source_module_of_properties: module, target_module_name: str) -> None:
    """
    Makes python properties in target module from functions found in the source module.
    
        :param source_module_of_properties: the source module from which functions should be taken.
        :param target_module_name: the name of the module to which properties are added.
        
    """
