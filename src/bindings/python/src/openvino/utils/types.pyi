# type: ignore
"""
Functions related to converting between Python and numpy types and openvino types.
"""
from __future__ import annotations
from openvino.exceptions import OVTypeError
from openvino._pyopenvino import Node
from openvino._pyopenvino import Output
from openvino._pyopenvino import Shape
from openvino._pyopenvino import Type
from openvino._pyopenvino.op import Constant
import logging as logging
import numpy
import numpy as np
import openvino._pyopenvino
import openvino._pyopenvino.op
import typing
__all__ = ['Constant', 'Node', 'NodeInput', 'NumericData', 'NumericType', 'OVTypeError', 'Output', 'ScalarData', 'Shape', 'TensorShape', 'Type', 'as_node', 'as_nodes', 'get_dtype', 'get_element_type', 'get_element_type_str', 'get_ndarray', 'get_numpy_ctype', 'get_shape', 'log', 'logging', 'make_constant_node', 'np', 'openvino_to_numpy_types_map', 'openvino_to_numpy_types_str_map']
def as_node(input_value: typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray], name: typing.Optional[str] = None) -> openvino._pyopenvino.Node:
    """
    Return input values as nodes. Scalars will be converted to Constant nodes.
    """
def as_nodes(*input_values: typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray], name: typing.Optional[str] = None) -> typing.List[openvino._pyopenvino.Node]:
    """
    Return input values as nodes. Scalars will be converted to Constant nodes.
    """
def get_dtype(openvino_type: openvino._pyopenvino.Type) -> numpy.dtype:
    """
    Return a numpy.dtype for an openvino element type.
    """
def get_element_type(data_type: typing.Union[type, numpy.dtype]) -> openvino._pyopenvino.Type:
    """
    Return an ngraph element type for a Python type or numpy.dtype.
    """
def get_element_type_str(data_type: typing.Union[type, numpy.dtype]) -> str:
    """
    Return an ngraph element type string representation for a Python type or numpy dtype.
    """
def get_ndarray(data: typing.Union[int, float, numpy.ndarray]) -> numpy.ndarray:
    """
    Wrap data into a numpy ndarray.
    """
def get_numpy_ctype(openvino_type: openvino._pyopenvino.Type) -> type:
    """
    Return numpy ctype for an openvino element type.
    """
def get_shape(data: typing.Union[int, float, numpy.ndarray]) -> typing.List[int]:
    """
    Return a shape of NumericData.
    """
def make_constant_node(value: typing.Union[int, float, numpy.ndarray], dtype: typing.Union[type, numpy.dtype, openvino._pyopenvino.Type] = None, *, name: typing.Optional[str] = None) -> openvino._pyopenvino.op.Constant:
    """
    Return an openvino Constant node with the specified value.
    """
NodeInput: typing._UnionGenericAlias  # value = typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray]
NumericData: typing._UnionGenericAlias  # value = typing.Union[int, float, numpy.ndarray]
NumericType: typing._UnionGenericAlias  # value = typing.Union[type, numpy.dtype]
ScalarData: typing._UnionGenericAlias  # value = typing.Union[int, float]
TensorShape: typing._GenericAlias  # value = typing.List[int]
log: logging.Logger  # value = <Logger openvino.utils.types (INFO)>
openvino_to_numpy_types_map: list  # value = [(<Type: 'char'>, bool), (<Type: 'char'>, numpy.bool), (<Type: 'float16'>, numpy.float16), (<Type: 'float32'>, numpy.float32), (<Type: 'double64'>, numpy.float64), (<Type: 'int8_t'>, numpy.int8), (<Type: 'int16_t'>, numpy.int16), (<Type: 'int32_t'>, numpy.int32), (<Type: 'int64_t'>, numpy.int64), (<Type: 'uint8_t'>, numpy.uint8), (<Type: 'uint16_t'>, numpy.uint16), (<Type: 'uint32_t'>, numpy.uint32), (<Type: 'uint64_t'>, numpy.uint64), (<Type: 'bfloat16'>, numpy.uint16), (<Type: 'string'>, str), (<Type: 'string'>, numpy.str_), (<Type: 'string'>, bytes), (<Type: 'string'>, numpy.bytes_)]
openvino_to_numpy_types_str_map: list = [('boolean', bool), ('boolean', numpy.bool), ('f16', numpy.float16), ('f32', numpy.float32), ('f64', numpy.float64), ('i8', numpy.int8), ('i16', numpy.int16), ('i32', numpy.int32), ('i64', numpy.int64), ('u8', numpy.uint8), ('u16', numpy.uint16), ('u32', numpy.uint32), ('u64', numpy.uint64), ('string', str), ('string', numpy.str_), ('string', bytes), ('string', numpy.bytes_)]
