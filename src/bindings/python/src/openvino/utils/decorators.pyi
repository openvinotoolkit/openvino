# type: ignore
from __future__ import annotations
from collections.abc import Callable
from functools import wraps
from inspect import signature
from openvino._pyopenvino import Node
from openvino._pyopenvino import Output
from openvino.utils.types import as_node
from openvino.utils.types import as_nodes
from typing import get_args
from typing import get_origin
import collections.abc
import openvino._pyopenvino
import typing
__all__: list[str] = ['Callable', 'MultiMethod', 'Node', 'NodeInput', 'Output', 'as_node', 'as_nodes', 'binary_op', 'custom_preprocess_function', 'get_args', 'get_origin', 'nameable_op', 'overloading', 'registry', 'signature', 'unary_op', 'wraps']
class MultiMethod:
    def __call__(self, *args, **kwargs) -> typing.Any:
        ...
    def __init__(self, name: str):
        ...
    def check_invoked_types_in_overloaded_funcs(self, tuple_to_check: tuple, key_structure: tuple) -> bool:
        ...
    def matches_optional(self, optional_type, actual_type) -> bool:
        ...
    def matches_union(self, union_type, actual_type) -> bool:
        ...
    def register(self, types: tuple, function: collections.abc.Callable) -> None:
        ...
def _get_name(**kwargs: typing.Any) -> openvino._pyopenvino.Node:
    ...
def _set_node_friendly_name(node: openvino._pyopenvino.Node, *, name: typing.Optional[str] = None) -> openvino._pyopenvino.Node:
    ...
def binary_op(node_factory_function: collections.abc.Callable) -> collections.abc.Callable:
    """
    Convert the first two input values to Constant Nodes if numeric values are detected.
    """
def custom_preprocess_function(custom_function: collections.abc.Callable) -> collections.abc.Callable:
    """
    Convert Node returned from custom_function to Output.
    """
def nameable_op(node_factory_function: collections.abc.Callable) -> collections.abc.Callable:
    """
    Set the name to the openvino operator returned by the wrapped function.
    """
def overloading(*types: tuple) -> collections.abc.Callable:
    ...
def unary_op(node_factory_function: collections.abc.Callable) -> collections.abc.Callable:
    """
    Convert the first input value to a Constant Node if a numeric value is detected.
    """
NodeInput: typing._UnionGenericAlias  # value = typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray]
registry: dict  # value = {'read_value': <openvino.utils.decorators.MultiMethod object>, 'constant': <openvino.utils.decorators.MultiMethod object>}
