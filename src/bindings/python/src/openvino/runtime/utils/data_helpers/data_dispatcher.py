# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from functools import singledispatch
from typing import Any, Dict, Union, Optional

import numpy as np

from openvino._pyopenvino import ConstOutput, Tensor, Type
from openvino.runtime.utils.data_helpers.wrappers import _InferRequestWrapper

ContainerTypes = Union[dict, list, tuple]
ScalarTypes = Union[np.number, int, float]
ValidKeys = Union[str, int, ConstOutput]


def get_request_tensor(
    request: _InferRequestWrapper,
    key: Optional[ValidKeys] = None,
) -> Tensor:
    if key is None:
        return request.get_input_tensor()
    elif isinstance(key, int):
        return request.get_input_tensor(key)
    elif isinstance(key, (str, ConstOutput)):
        return request.get_tensor(key)
    else:
        raise TypeError(f"Unsupported key type: {type(key)} for Tensor under key: {key}")


@singledispatch
def value_to_tensor(
    value: Union[Tensor, np.ndarray, ScalarTypes],
    request: Optional[_InferRequestWrapper] = None,
    is_shared: bool = False,
    key: Optional[ValidKeys] = None,
) -> None:
    raise TypeError(f"Incompatible inputs of type: {type(value)}")


@value_to_tensor.register(Tensor)
def _(
    value: Tensor,
    request: Optional[_InferRequestWrapper] = None,
    is_shared: bool = False,
    key: Optional[ValidKeys] = None,
) -> Tensor:
    return value


@value_to_tensor.register(np.ndarray)
def _(
    value: np.ndarray,
    request: _InferRequestWrapper,
    is_shared: bool = False,
    key: Optional[ValidKeys] = None,
) -> Tensor:
    tensor = get_request_tensor(request, key)
    tensor_type = tensor.get_element_type()
    tensor_dtype = tensor_type.to_dtype()
    if value.ndim == 0:
        tensor_shape = tuple(tensor.shape)
        if tensor_dtype == value.dtype and tensor_shape == value.shape:
            return Tensor(value, shared_memory=is_shared)
        else:
            return Tensor(value.astype(tensor_dtype).reshape(tensor_shape), shared_memory=False)
    # WA for FP16-->BF16 edge-case, always copy.
    if tensor_type == Type.bf16:
        tensor = Tensor(tensor_type, value.shape)
        tensor.data[:] = value.view(tensor_dtype)
        return tensor
    # WA for "not writeable" edge-case, always copy.
    if value.flags["WRITEABLE"] is False:
        tensor = Tensor(tensor_type, value.shape)
        tensor.data[:] = value.astype(tensor_dtype) if tensor_dtype != value.dtype else value
        return tensor
    # If types are mismatched, convert and always copy.
    if tensor_dtype != value.dtype:
        return Tensor(value.astype(tensor_dtype), shared_memory=False)
    # Otherwise, use mode defined in the call.
    return Tensor(value, shared_memory=is_shared)


@value_to_tensor.register(np.number)
@value_to_tensor.register(int)
@value_to_tensor.register(float)
def _(
    value: ScalarTypes,
    request: _InferRequestWrapper,
    is_shared: bool = False,
    key: Optional[ValidKeys] = None,
) -> Tensor:
    # np.number/int/float edge-case, copy will occur in both scenarios.
    tensor_type = get_request_tensor(request, key).get_element_type()
    tensor_dtype = tensor_type.to_dtype()
    tmp = np.array(value)
    # If types are mismatched, convert.
    if tensor_dtype != tmp.dtype:
        return Tensor(tmp.astype(tensor_dtype), shared_memory=False)
    return Tensor(tmp, shared_memory=False)


def to_c_style(value: Any, is_shared: bool = False) -> Any:
    if not isinstance(value, np.ndarray):
        if hasattr(value, "__array__"):
            return to_c_style(np.array(value, copy=False)) if is_shared else np.array(value, copy=True)
        return value
    return value if value.flags["C_CONTIGUOUS"] else np.ascontiguousarray(value)


###
# Start of array normalization.
###
@singledispatch
def normalize_arrays(
    inputs: Any,
    is_shared: bool = False,
) -> Any:
    # Check the special case of the array-interface
    if hasattr(inputs, "__array__"):
        return to_c_style(np.array(inputs, copy=False)) if is_shared else np.array(inputs, copy=True)
    # Error should be raised if type does not match any dispatchers
    raise TypeError(f"Incompatible inputs of type: {type(inputs)}")


@normalize_arrays.register(dict)
def _(
    inputs: dict,
    is_shared: bool = False,
) -> dict:
    return {k: to_c_style(v) if is_shared else v for k, v in inputs.items()}


@normalize_arrays.register(list)
@normalize_arrays.register(tuple)
def _(
    inputs: Union[list, tuple],
    is_shared: bool = False,
) -> dict:
    return {i: to_c_style(v) if is_shared else v for i, v in enumerate(inputs)}


@normalize_arrays.register(np.ndarray)
def _(
    inputs: dict,
    is_shared: bool = False,
) -> Any:
    return to_c_style(inputs) if is_shared else inputs
###
# End of array normalization.
###


###
# Start of "shared" dispatcher.
# (1) Each method should keep Tensors "as-is", regardless to them being shared or not.
# (2) ...
###
# Step to keep alive input values that are not C-style by default
@singledispatch
def create_shared(
    inputs: Any,
    request: _InferRequestWrapper,
) -> None:
    # Check the special case of the array-interface
    if hasattr(inputs, "__array__"):
        request._inputs_data = normalize_arrays(inputs, is_shared=True)
        return value_to_tensor(request._inputs_data, request=request, is_shared=True)
    # Error should be raised if type does not match any dispatchers
    raise TypeError(f"Incompatible inputs of type: {type(inputs)}")


@create_shared.register(dict)
@create_shared.register(list)
@create_shared.register(tuple)
def _(
    inputs: ContainerTypes,
    request: _InferRequestWrapper,
) -> dict:
    request._inputs_data = normalize_arrays(inputs, is_shared=True)
    return {k: value_to_tensor(v, request=request, is_shared=True, key=k) for k, v in request._inputs_data.items()}


@create_shared.register(np.ndarray)
def _(
    inputs: np.ndarray,
    request: _InferRequestWrapper,
) -> Tensor:
    request._inputs_data = normalize_arrays(inputs, is_shared=True)
    return value_to_tensor(request._inputs_data, request=request, is_shared=True)


@create_shared.register(Tensor)
@create_shared.register(np.number)
@create_shared.register(int)
@create_shared.register(float)
def _(
    inputs: Union[Tensor, ScalarTypes],
    request: _InferRequestWrapper,
) -> Tensor:
    return value_to_tensor(inputs, request=request, is_shared=True)
###
# End of "shared" dispatcher methods.
###


###
# Start of "copied" dispatcher.
###
def set_request_tensor(
    request: _InferRequestWrapper,
    tensor: Tensor,
    key: Optional[ValidKeys] = None,
) -> None:
    if key is None:
        request.set_input_tensor(tensor)
    elif isinstance(key, int):
        request.set_input_tensor(key, tensor)
    elif isinstance(key, (str, ConstOutput)):
        request.set_tensor(key, tensor)
    else:
        raise TypeError(f"Unsupported key type: {type(key)} for Tensor under key: {key}")


@singledispatch
def update_tensor(
    inputs: Any,
    request: _InferRequestWrapper,
    key: Optional[ValidKeys] = None,
) -> None:
    if hasattr(inputs, "__array__"):
        update_tensor(normalize_arrays(inputs, is_shared=False), request, key)
        return None
    raise TypeError(f"Incompatible inputs of type: {type(inputs)} under {key} key!")


@update_tensor.register(np.ndarray)
def _(
    inputs: np.ndarray,
    request: _InferRequestWrapper,
    key: Optional[ValidKeys] = None,
) -> None:
    if inputs.ndim != 0:
        tensor = get_request_tensor(request, key)
        # Update shape if there is a mismatch
        if tuple(tensor.shape) != inputs.shape:
            tensor.shape = inputs.shape
        # When copying, type should be up/down-casted automatically.
        tensor.data[:] = inputs[:]
    else:
        # If shape is "empty", assume this is a scalar value
        set_request_tensor(
            request,
            value_to_tensor(inputs, request=request, is_shared=False, key=key),
            key,
        )


@update_tensor.register(np.number)  # type: ignore
@update_tensor.register(float)
@update_tensor.register(int)
def _(
    inputs: Union[np.number, float, int],
    request: _InferRequestWrapper,
    key: Optional[ValidKeys] = None,
) -> None:
    set_request_tensor(
        request,
        value_to_tensor(inputs, request=request, is_shared=False, key=key),
        key,
    )


def update_inputs(inputs: dict, request: _InferRequestWrapper) -> dict:
    """Helper function to prepare inputs for inference.

    It creates copy of Tensors or copy data to already allocated Tensors on device
    if the item is of type `np.ndarray`, `np.number`, `int`, `float` or has numpy __array__ attribute.
    """
    # Create new temporary dictionary.
    # new_inputs will be used to transfer data to inference calls,
    # ensuring that original inputs are not overwritten with Tensors.
    new_inputs: Dict[ValidKeys, Tensor] = {}
    for key, value in inputs.items():
        if not isinstance(key, (str, int, ConstOutput)):
            raise TypeError(f"Incompatible key type for input: {key}")
        # Copy numpy arrays to already allocated Tensors.
        # If value object has __array__ attribute, load it to Tensor using np.array
        if isinstance(value, (np.ndarray, np.number, int, float)) or hasattr(value, "__array__"):
            update_tensor(value, request, key)
        # If value is of Tensor type, put it into temporary dictionary.
        elif isinstance(value, Tensor):
            new_inputs[key] = value
        # Throw error otherwise.
        else:
            raise TypeError(f"Incompatible inputs of type: {type(value)} under {key} key!")
    return new_inputs


@singledispatch
def create_copied(
    inputs: Union[ContainerTypes, np.ndarray, ScalarTypes],
    request: _InferRequestWrapper,
) -> Union[dict, None]:
    # Check the special case of the array-interface
    if hasattr(inputs, "__array__"):
        update_tensor(normalize_arrays(inputs, is_shared=False), request, key=None)
        return {}
    # Error should be raised if type does not match any dispatchers
    raise TypeError(f"Incompatible inputs of type: {type(inputs)}")


@create_copied.register(dict)
@create_copied.register(list)
@create_copied.register(tuple)
def _(
    inputs: ContainerTypes,
    request: _InferRequestWrapper,
) -> dict:
    return update_inputs(normalize_arrays(inputs, is_shared=False), request)


@create_copied.register(np.ndarray)
def _(
    inputs: np.ndarray,
    request: _InferRequestWrapper,
) -> dict:
    update_tensor(normalize_arrays(inputs, is_shared=False), request, key=None)
    return {}


@create_copied.register(Tensor)
@create_copied.register(np.number)
@create_copied.register(int)
@create_copied.register(float)
def _(
    inputs: Union[Tensor, ScalarTypes],
    request: _InferRequestWrapper,
) -> Tensor:
    return value_to_tensor(inputs, request=request, is_shared=False)
###
# End of "copied" dispatcher methods.
###


def _data_dispatch(
    request: _InferRequestWrapper,
    inputs: Union[ContainerTypes, Tensor, np.ndarray, ScalarTypes] = None,
    is_shared: bool = False,
) -> Union[dict, Tensor]:
    if inputs is None:
        return {}
    return create_shared(inputs, request) if is_shared else create_copied(inputs, request)
