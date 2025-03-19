# type: ignore
from functools import singledispatch
from __future__ import annotations
from openvino._pyopenvino import ConstOutput
from openvino._pyopenvino import RemoteTensor
from openvino._pyopenvino import Tensor
from openvino._pyopenvino import Type
from openvino.utils.data_helpers.wrappers import _InferRequestWrapper
from openvino.utils.data_helpers.wrappers import OVDict
from typing import Any
import numpy as np
import openvino._pyopenvino
import openvino.utils.data_helpers.wrappers
import typing
__all__ = ['Any', 'ConstOutput', 'ContainerTypes', 'OVDict', 'RemoteTensor', 'ScalarTypes', 'Tensor', 'Type', 'ValidKeys', 'create_copied', 'create_shared', 'get_request_tensor', 'is_list_simple_type', 'normalize_arrays', 'np', 'set_request_tensor', 'singledispatch', 'to_c_style', 'update_inputs', 'update_tensor', 'value_to_tensor']
def _(inputs: typing.Union[openvino._pyopenvino.Tensor, numpy.number, int, float, str, bytes], request: openvino.utils.data_helpers.wrappers._InferRequestWrapper) -> openvino._pyopenvino.Tensor:
    ...
def _data_dispatch(request: openvino.utils.data_helpers.wrappers._InferRequestWrapper, inputs: typing.Union[dict, list, tuple, openvino.utils.data_helpers.wrappers.OVDict, openvino._pyopenvino.Tensor, numpy.ndarray, numpy.number, int, float, str] = None, is_shared: bool = False) -> typing.Union[dict, openvino._pyopenvino.Tensor]:
    ...
def create_copied(*args, **kw) -> typing.Optional[dict]:
    ...
def create_shared(*args, **kw) -> None:
    ...
def get_request_tensor(request: openvino.utils.data_helpers.wrappers._InferRequestWrapper, key: typing.Union[str, int, openvino._pyopenvino.ConstOutput, NoneType] = None) -> openvino._pyopenvino.Tensor:
    ...
def is_list_simple_type(input_list: list) -> bool:
    ...
def normalize_arrays(*args, **kw) -> typing.Any:
    ...
def set_request_tensor(request: openvino.utils.data_helpers.wrappers._InferRequestWrapper, tensor: openvino._pyopenvino.Tensor, key: typing.Union[str, int, openvino._pyopenvino.ConstOutput, NoneType] = None) -> None:
    ...
def to_c_style(value: typing.Any, is_shared: bool = False) -> typing.Any:
    ...
def update_inputs(inputs: dict, request: openvino.utils.data_helpers.wrappers._InferRequestWrapper) -> dict:
    """
    Helper function to prepare inputs for inference.
    
        It creates copy of Tensors or copy data to already allocated Tensors on device
        if the item is of type `np.ndarray`, `np.number`, `int`, `float` or has numpy __array__ attribute.
        If value is of type `list`, create a Tensor based on it, copy will occur in the Tensor constructor.
        
    """
def update_tensor(*args, **kw) -> None:
    ...
def value_to_tensor(*args, **kw) -> None:
    ...
ContainerTypes: typing._UnionGenericAlias  # value = typing.Union[dict, list, tuple, openvino.utils.data_helpers.wrappers.OVDict]
ScalarTypes: typing._UnionGenericAlias  # value = typing.Union[numpy.number, int, float]
ValidKeys: typing._UnionGenericAlias  # value = typing.Union[str, int, openvino._pyopenvino.ConstOutput]
