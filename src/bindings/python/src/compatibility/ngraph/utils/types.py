# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Functions related to converting between Python and numpy types and ngraph types."""

import logging
from typing import List, Union

import numpy as np

from ngraph.exceptions import NgraphTypeError
from ngraph.impl import Node, Shape, Output
from ngraph.impl import Type as NgraphType
from ngraph.impl.op import Constant

log = logging.getLogger(__name__)

TensorShape = List[int]
NumericData = Union[int, float, np.ndarray]
NumericType = Union[type, np.dtype]
ScalarData = Union[int, float]
NodeInput = Union[Node, NumericData]

ngraph_to_numpy_types_map = [
    (NgraphType.boolean, np.bool),
    (NgraphType.f16, np.float16),
    (NgraphType.f32, np.float32),
    (NgraphType.f64, np.float64),
    (NgraphType.i8, np.int8),
    (NgraphType.i16, np.int16),
    (NgraphType.i32, np.int32),
    (NgraphType.i64, np.int64),
    (NgraphType.u8, np.uint8),
    (NgraphType.u16, np.uint16),
    (NgraphType.u32, np.uint32),
    (NgraphType.u64, np.uint64),
    (NgraphType.bf16, np.uint16),
]

ngraph_to_numpy_types_str_map = [
    ("boolean", np.bool),
    ("f16", np.float16),
    ("f32", np.float32),
    ("f64", np.float64),
    ("i8", np.int8),
    ("i16", np.int16),
    ("i32", np.int32),
    ("i64", np.int64),
    ("u8", np.uint8),
    ("u16", np.uint16),
    ("u32", np.uint32),
    ("u64", np.uint64),
]


def get_element_type(data_type: NumericType) -> NgraphType:
    """Return an ngraph element type for a Python type or numpy.dtype."""
    if data_type is int:
        log.warning("Converting int type of undefined bitwidth to 32-bit ngraph integer.")
        return NgraphType.i32

    if data_type is float:
        log.warning("Converting float type of undefined bitwidth to 32-bit ngraph float.")
        return NgraphType.f32

    ng_type = next(
        (ng_type for (ng_type, np_type) in ngraph_to_numpy_types_map if np_type == data_type), None
    )
    if ng_type:
        return ng_type

    raise NgraphTypeError("Unidentified data type %s", data_type)


def get_element_type_str(data_type: NumericType) -> str:
    """Return an ngraph element type string representation for a Python type or numpy dtype."""
    if data_type is int:
        log.warning("Converting int type of undefined bitwidth to 32-bit ngraph integer.")
        return "i32"

    if data_type is float:
        log.warning("Converting float type of undefined bitwidth to 32-bit ngraph float.")
        return "f32"

    ng_type = next(
        (ng_type for (ng_type, np_type) in ngraph_to_numpy_types_str_map if np_type == data_type),
        None,
    )
    if ng_type:
        return ng_type

    raise NgraphTypeError("Unidentified data type %s", data_type)


def get_dtype(ngraph_type: NgraphType) -> np.dtype:
    """Return a numpy.dtype for an ngraph element type."""
    np_type = next(
        (np_type for (ng_type, np_type) in ngraph_to_numpy_types_map if ng_type == ngraph_type),
        None,
    )

    if np_type:
        return np.dtype(np_type)

    raise NgraphTypeError("Unidentified data type %s", ngraph_type)


def get_ndarray(data: NumericData) -> np.ndarray:
    """Wrap data into a numpy ndarray."""
    if type(data) == np.ndarray:
        return data
    return np.array(data)


def get_shape(data: NumericData) -> TensorShape:
    """Return a shape of NumericData."""
    if type(data) == np.ndarray:
        return data.shape  # type: ignore
    elif type(data) == list:
        return [len(data)]  # type: ignore
    return []


def make_constant_node(value: NumericData, dtype: NumericType = None) -> Constant:
    """Return an ngraph Constant node with the specified value."""
    ndarray = get_ndarray(value)
    if dtype:
        element_type = get_element_type(dtype)
    else:
        element_type = get_element_type(ndarray.dtype)

    return Constant(element_type, Shape(ndarray.shape), ndarray.flatten().tolist())


def as_node(input_value: NodeInput) -> Node:
    """Return input values as nodes. Scalars will be converted to Constant nodes."""
    if issubclass(type(input_value), Node):
        return input_value
    if issubclass(type(input_value), Output):
        return input_value
    return make_constant_node(input_value)


def as_nodes(*input_values: NodeInput) -> List[Node]:
    """Return input values as nodes. Scalars will be converted to Constant nodes."""
    return [as_node(input_value) for input_value in input_values]
