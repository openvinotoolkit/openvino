import numpy as np
import openvino.utils.decorators
import typing
from functools import partial as partial, singledispatch as singledispatch
from openvino._pyopenvino import Node as Node, Output as Output, PartialShape as PartialShape, Shape as Shape, Type as Type
from openvino._pyopenvino.op import Constant as Constant, Parameter as Parameter, assign as assign
from openvino._pyopenvino.op.util import Variable as Variable, VariableInfo as VariableInfo
from openvino.utils.decorators import nameable_op as nameable_op, overloading as overloading
from openvino.utils.types import as_node as as_node, as_nodes as as_nodes, get_element_type as get_element_type

def ctc_greedy_decoder_seq_len(*args, **kwargs) -> openvino._pyopenvino.Node: ...
def gather_elements(*args, **kwargs) -> openvino._pyopenvino.Node: ...
def mvn(*args, **kwargs) -> openvino._pyopenvino.Node: ...

NodeInput: typing._UnionGenericAlias
NumericType: typing._UnionGenericAlias
TensorShape: typing._GenericAlias
read_value: openvino.utils.decorators.MultiMethod
