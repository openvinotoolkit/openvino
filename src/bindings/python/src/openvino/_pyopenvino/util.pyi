# type: ignore
"""
openvino.utils
"""
from __future__ import annotations
import numpy
import openvino._pyopenvino
import typing
__all__ = ['deprecation_warning', 'numpy_to_c', 'replace_node', 'replace_output_update_name']
def deprecation_warning(function_name: str, version: str = '', message: str = '', stacklevel: int = 2) -> None:
    """
                Prints deprecation warning "{function_name} is deprecated and will be removed in version {version}. {message}".
    
                :param function_name: The name of the deprecated function.
                :param version: The version in which the code will be removed.
                :param message: A message explaining why the function is deprecated.
                :param stacklevel: How many layers should be propagated.
    """
def numpy_to_c(arg0: numpy.ndarray[typing.Any, numpy.dtype[typing.Any]]) -> capsule:
    ...
@typing.overload
def replace_node(target: openvino._pyopenvino.Node, replacement: openvino._pyopenvino.Node) -> None:
    ...
@typing.overload
def replace_node(target: openvino._pyopenvino.Node, replacement: list[openvino._pyopenvino.Output]) -> None:
    ...
@typing.overload
def replace_node(target: openvino._pyopenvino.Node, replacement: openvino._pyopenvino.Node, outputs_order: list[int]) -> None:
    ...
def replace_output_update_name(output: openvino._pyopenvino.Output, target_output: openvino._pyopenvino.Output) -> bool:
    ...
