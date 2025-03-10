import numpy as np
import openvino._pyopenvino
import typing
from functools import partial as partial
from openvino._pyopenvino import Node as Node, Shape as Shape
from openvino._pyopenvino.op import Constant as Constant, Parameter as Parameter
from openvino.utils.decorators import binary_op as binary_op, nameable_op as nameable_op, unary_op as unary_op
from openvino.utils.input_validation import assert_list_of_ints as assert_list_of_ints, check_valid_attributes as check_valid_attributes, is_non_negative_value as is_non_negative_value, is_positive_value as is_positive_value
from openvino.utils.node_factory import NodeFactory as NodeFactory
from openvino.utils.types import as_node as as_node, as_nodes as as_nodes, get_dtype as get_dtype, get_element_type as get_element_type, get_element_type_str as get_element_type_str, make_constant_node as make_constant_node

def dft(data: typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray], axes: typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray], signal_size: typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray, NoneType] = ..., name: typing.Optional[str] = ...) -> openvino._pyopenvino.Node: ...
def einsum(*args, **kwargs) -> openvino._pyopenvino.Node: ...
def gather(*args, **kwargs) -> openvino._pyopenvino.Node: ...
def gelu(*args, **kwargs) -> openvino._pyopenvino.Node: ...
def idft(*args, **kwargs) -> openvino._pyopenvino.Node: ...
def roll(*args, **kwargs) -> openvino._pyopenvino.Node: ...

NodeInput: typing._UnionGenericAlias
NumericData: typing._UnionGenericAlias
NumericType: typing._UnionGenericAlias
ScalarData: typing._UnionGenericAlias
TensorShape: typing._GenericAlias
