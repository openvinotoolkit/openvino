from __future__ import annotations
import numpy
from openvino.utils.types import as_node
from openvino.utils.types import as_nodes
from openvino.utils.types import get_dtype
from openvino.utils.types import get_element_type
from openvino.utils.types import get_element_type_str
from openvino.utils.types import get_ndarray
from openvino.utils.types import get_numpy_ctype
from openvino.utils.types import get_shape
from openvino.utils.types import make_constant_node
import typing
__all__ = ['NodeInput', 'NumericData', 'NumericType', 'ScalarData', 'TensorShape', 'as_node', 'as_nodes', 'get_dtype', 'get_element_type', 'get_element_type_str', 'get_ndarray', 'get_numpy_ctype', 'get_shape', 'make_constant_node', 'openvino_to_numpy_types_map', 'openvino_to_numpy_types_str_map']
NodeInput: typing._UnionGenericAlias  # value = typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray]
NumericData: typing._UnionGenericAlias  # value = typing.Union[int, float, numpy.ndarray]
NumericType: typing._UnionGenericAlias  # value = typing.Union[type, numpy.dtype]
ScalarData: typing._UnionGenericAlias  # value = typing.Union[int, float]
TensorShape: typing._GenericAlias  # value = typing.List[int]
openvino_to_numpy_types_map: list  # value = [(<Type: 'char'>, bool), (<Type: 'char'>, numpy.bool), (<Type: 'float16'>, numpy.float16), (<Type: 'float32'>, numpy.float32), (<Type: 'double64'>, numpy.float64), (<Type: 'int8_t'>, numpy.int8), (<Type: 'int16_t'>, numpy.int16), (<Type: 'int32_t'>, numpy.int32), (<Type: 'int64_t'>, numpy.int64), (<Type: 'uint8_t'>, numpy.uint8), (<Type: 'uint16_t'>, numpy.uint16), (<Type: 'uint32_t'>, numpy.uint32), (<Type: 'uint64_t'>, numpy.uint64), (<Type: 'bfloat16'>, numpy.uint16), (<Type: 'string'>, str), (<Type: 'string'>, numpy.str_), (<Type: 'string'>, bytes), (<Type: 'string'>, numpy.bytes_)]
openvino_to_numpy_types_str_map: list = [('boolean', bool), ('boolean', numpy.bool), ('f16', numpy.float16), ('f32', numpy.float32), ('f64', numpy.float64), ('i8', numpy.int8), ('i16', numpy.int16), ('i32', numpy.int32), ('i64', numpy.int64), ('u8', numpy.uint8), ('u16', numpy.uint16), ('u32', numpy.uint32), ('u64', numpy.uint64), ('string', str), ('string', numpy.str_), ('string', bytes), ('string', numpy.bytes_)]
