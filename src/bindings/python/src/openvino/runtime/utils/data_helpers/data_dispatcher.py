from functools import singledispatch
from typing import Dict, Union

import numpy as np

from openvino._pyopenvino import ConstOutput, Tensor
from openvino._pyopenvino import InferRequest as InferRequestBase


class InferRequestInternal(InferRequestBase):
    """InferRequest class with internal memory."""

    def __init__(self, other):
        # Private memeber to store newly created shared memory data
        self._inputs_data = None
        super().__init__(other)


def value_to_tensor(val, is_shared: bool = False):
    # Special case for Tensor, return "as-is"
    if isinstance(val, Tensor):
        return val
    elif isinstance(val, (np.ndarray, np.number, int, float)):
        # Edge-case for numpy arrays if shape is "empty",
        # assume this is a scalar value - always copy
        if not val.shape or isinstance(val, (np.number, int, float)):
            return Tensor(np.ndarray([], type(val), np.array(val)))
        return Tensor(val, shared_memory=is_shared)
    else:
        raise TypeError(f"Incompatible inputs of type: {type(val)}")


def to_c_style(val, is_shared: bool = False):
    if not isinstance(val, np.ndarray):
        if hasattr(val, "__array__"):
            return to_c_style(np.array(val, copy=False)) if is_shared else np.array(val, copy=True)
        return val
    # Check C-style if not convert data (or raise error?)
    return val if val.flags["C_CONTIGUOUS"] else np.ascontiguousarray(val)


###
# Start of array normalization.
###
@singledispatch
def normalize_arrays(
    inputs: Union[
        dict,
        list,
        tuple,
        np.ndarray,
    ],
    is_shared: bool = False,
) -> None:
    # Check the special case of the array-interface
    if hasattr(inputs, "__array__"):
        return to_c_style(np.array(inputs, copy=False)) if is_shared else np.array(inputs, copy=True)
    # Error should be raised if type does not match any dispatchers
    raise TypeError(f"Incompatible inputs of type: {type(inputs)}")


@normalize_arrays.register(dict)
def _(
    inputs: dict,
    is_shared: bool = False,
):
    return {k: to_c_style(v) if is_shared else v for k, v in inputs.items()}


@normalize_arrays.register(list)
@normalize_arrays.register(tuple)
def _(
    inputs: Union[list, tuple],
    is_shared: bool = False,
):
    return {i: to_c_style(v) if is_shared else v for i, v in enumerate(inputs)}


@normalize_arrays.register(np.ndarray)
def _(
    inputs: dict,
    is_shared: bool = False,
):
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
    inputs: Union[np.ndarray, np.number, int, float],
    request: InferRequestInternal,
) -> None:
    # Check the special case of the array-interface
    if hasattr(inputs, "__array__"):
        request._inputs_data = normalize_arrays(inputs, is_shared=True)
        return value_to_tensor(request._inputs_data, is_shared=True)
    # Error should be raised if type does not match any dispatchers
    raise TypeError(f"Incompatible inputs of type: {type(inputs)}")


@create_shared.register(dict)
@create_shared.register(list)
@create_shared.register(tuple)
def _(
    inputs: Union[dict, list, tuple],
    request: InferRequestInternal,
) -> None:
    request._inputs_data = normalize_arrays(inputs, is_shared=True)
    return {k: value_to_tensor(v, is_shared=True) for k, v in request._inputs_data.items()}


@create_shared.register(np.ndarray)
def _(
    inputs: np.ndarray,
    request: InferRequestInternal,
) -> None:
    request._inputs_data = normalize_arrays(inputs, is_shared=True)
    return value_to_tensor(request._inputs_data, is_shared=True)


@create_shared.register(Tensor)
@create_shared.register(np.number)
@create_shared.register(int)
@create_shared.register(float)
def _(
    inputs: Union[Tensor, np.number, float, int],
    request: InferRequestInternal,
) -> None:
    # Special case
    return value_to_tensor(inputs)


###
# End of "shared" dispatcher methods.
###

###
# Start of "copied" dispatcher.
###
def set_request_tensor(request: InferRequestInternal, tensor: Tensor, key: Union[str, int, ConstOutput] = None) -> None:
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
    inputs: Union[np.ndarray, np.number, int, float],
    request: InferRequestInternal,
    key: Union[str, int, ConstOutput] = None,
) -> None:
    if hasattr(inputs, "__array__"):
        update_tensor(normalize_arrays(inputs, is_shared=False), request, key=None)
        return None
    raise TypeError(f"Incompatible inputs of type: {type(inputs)} under {key} key!")


@update_tensor.register(np.ndarray)
def _(
    inputs: np.ndarray,
    request: InferRequestInternal,
    key: Union[str, int, ConstOutput] = None,
) -> None:
    # If shape is "empty", assume this is a scalar value
    if not inputs.shape:
        set_request_tensor(request, Tensor(inputs), key)
    else:
        if key is None:
            tensor = request.get_input_tensor()
        elif isinstance(key, int):
            tensor = request.get_input_tensor(key)
        elif isinstance(key, (str, ConstOutput)):
            tensor = request.get_tensor(key)
        else:
            raise TypeError(f"Unsupported key type: {type(key)} for Tensor under key: {key}")
        # Update shape if there is a mismatch
        if tensor.shape != inputs.shape:
            tensor.shape = inputs.shape
        # When copying, type should be up/down-casted automatically.
        tensor.data[:] = inputs[:]


@update_tensor.register(np.number)  # type: ignore
@update_tensor.register(float)
@update_tensor.register(int)
def _(
    inputs: Union[np.number, float, int],
    request: InferRequestInternal,
    key: Union[str, int, ConstOutput] = None,
) -> None:
    set_request_tensor(
        request,
        value_to_tensor(inputs, is_shared=False),
        key,
    )


def update_inputs(inputs: dict, request: InferRequestInternal) -> dict:
    """Helper function to prepare inputs for inference.

    It creates copy of Tensors or copy data to already allocated Tensors on device
    if the item is of type `np.ndarray`, `np.number`, `int`, `float` or has numpy __array__ attribute.
    """
    # Create new temporary dictionary.
    # new_inputs will be used to transfer data to inference calls,
    # ensuring that original inputs are not overwritten with Tensors.
    new_inputs: Dict[Union[str, int, ConstOutput], Tensor] = {}
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
    inputs: Union[dict, list, tuple, np.ndarray, np.number, int, float],
    request: InferRequestInternal,
) -> None:
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
    inputs: Union[dict, list, tuple],
    request: InferRequestInternal,
) -> None:
    return update_inputs(normalize_arrays(inputs, is_shared=False), request)


@create_copied.register(np.ndarray)
def _(
    inputs: np.ndarray,
    request: InferRequestInternal,
) -> None:
    update_tensor(normalize_arrays(inputs, is_shared=False), request, key=None)
    return {}


@create_copied.register(Tensor)
@create_copied.register(np.number)
@create_copied.register(int)
@create_copied.register(float)
def _(
    inputs: Union[Tensor, np.number, float, int],
    request: InferRequestInternal,
) -> None:
    return value_to_tensor(inputs, is_shared=False)


###
# End of "copied" dispatcher methods.
###


def _data_dispatch(
    request: InferRequestInternal,
    inputs: Union[dict, list, tuple, Tensor, np.ndarray, np.number, int, float] = None,
    is_shared: bool = False,
):
    if inputs is None:
        return {}
    return create_shared(inputs, request) if is_shared else create_copied(inputs, request)
