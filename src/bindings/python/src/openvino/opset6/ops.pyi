# type: ignore
from functools import partial
from functools import singledispatch
from __future__ import annotations
from openvino._pyopenvino import Node
from openvino._pyopenvino import Output
from openvino._pyopenvino import PartialShape
from openvino._pyopenvino import Shape
from openvino._pyopenvino import Type
from openvino._pyopenvino.op import assign
from openvino._pyopenvino.op import Constant
from openvino._pyopenvino.op import Parameter
from openvino._pyopenvino.op import read_value as _read_value
from openvino._pyopenvino.op.util import Variable
from openvino._pyopenvino.op.util import VariableInfo
from openvino.utils.decorators import nameable_op
from openvino.utils.decorators import overloading
from openvino.utils.node_factory import _get_node_factory
from openvino.utils.types import as_node
from openvino.utils.types import as_nodes
from openvino.utils.types import get_element_type
import functools
import numpy as np
import openvino._pyopenvino
import openvino.utils.decorators
import typing
__all__ = ['Constant', 'Node', 'NodeInput', 'NumericType', 'Output', 'Parameter', 'PartialShape', 'Shape', 'TensorShape', 'Type', 'Variable', 'VariableInfo', 'as_node', 'as_nodes', 'assign', 'ctc_greedy_decoder_seq_len', 'gather_elements', 'get_element_type', 'mvn', 'nameable_op', 'np', 'overloading', 'partial', 'read_value', 'singledispatch']
def ctc_greedy_decoder_seq_len(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs CTCGreedyDecoderSeqLen.
    
        :param data:            The input 3D tensor. Shape: [batch_size, seq_length, num_classes]
        :param sequence_length: Input 1D tensor with sequence length. Shape: [batch_size]
        :param blank_index:     Scalar or 1D tensor with specifies the class index to use for the blank class.
                                Optional parameter. Default value is num_classes-1.
        :return:                The new node which performs CTCGreedyDecoderSeqLen.
        
    """
def gather_elements(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs GatherElements.
    
        :param data:       N-D tensor with data for gathering
        :param indices:    N-D tensor with indices by which data is gathered
        :param axis:       axis along which elements are gathered
        :return:           The new node which performs GatherElements
        
    """
def mvn(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs MeanVarianceNormalization (MVN).
    
        :param data: The node with data tensor.
        :param axes: The node with axes to reduce on.
        :param normalize_variance: Denotes whether to perform variance normalization.
        :param eps: The number added to the variance to avoid division by zero
                   when normalizing the value. Scalar value.
        :param eps_mode: how eps is applied (`inside_sqrt` or `outside_sqrt`)
        :param name: Optional output node name.
        :return: The new node performing a MVN operation on input tensor.
        
    """
NodeInput: typing._UnionGenericAlias  # value = typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray]
NumericType: typing._UnionGenericAlias  # value = typing.Union[type, numpy.dtype]
TensorShape: typing._GenericAlias  # value = typing.List[int]
_get_node_factory_opset6: functools.partial  # value = functools.partial(<function _get_node_factory at memory_address>, 'opset6')
read_value: openvino.utils.decorators.MultiMethod  # value = <openvino.utils.decorators.MultiMethod object>
