"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging as log

import numpy as np

from mo.front.common.partial_infer.eltwise import eltwise_infer, bias_add_infer
from mo.graph.graph import Graph
from mo.middle.passes.infer import copy_type_infer
from mo.ops.op import Op
from mo.pipeline.common import convert_const_node_value_type


class Elementwise(Op):
    enabled = False
    operation = None
    op = None
    op_type = None

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': self.op_type,
            'version': 'opset1',
            'infer': lambda node: eltwise_infer(node, self.operation),
            'type_infer': self.type_infer,
            'can_be_bias': True,
            'can_be_fused': True,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'is_eltwise': True,
            'stop_value_propagation': False
        }, attrs)

    @staticmethod
    def type_infer(node):
        in_type_0 = node.in_port(0).get_data_type()
        in_type_1 = node.in_port(1).get_data_type()
        if in_type_0 != in_type_1:
            # in case of input values data type mismatch we try to change the type of the constant to match the type of
            # another input. The input values data type mismatch occur when the MO performs replacement of some
            # operations like SquaredDifference of inputs with floating point data type to Power layer with the integer
            # power value 2, or when replacing Neg operation with Mul with -1 as second input.
            in_node_0 = node.in_port(0).get_source().node
            in_node_1 = node.in_port(1).get_source().node
            if in_node_0.op == 'Const':
                convert_const_node_value_type(in_node_0, in_type_1)
            elif in_node_1.op == 'Const':
                convert_const_node_value_type(in_node_1, in_type_0)
            else:
                log.error('Elementwise operation {} has inputs of different data types: {} and {}'.format(
                    node.soft_get('name'), in_type_0, in_type_1))

        node.out_port(0).set_data_type(node.in_port(0).get_data_type())


class Add(Elementwise):
    enabled = False
    op = 'Add'
    op_type = 'Add'
    operation = staticmethod(lambda a, b: a + b)


class BiasAdd(Add):
    op_type = 'BiasAdd'

    def __init__(self, graph: Graph, attrs: dict):
        attrs.update({'infer': lambda node: bias_add_infer(node, self.operation)})
        super().__init__(graph, attrs)


class Sub(Elementwise):
    enabled = False
    op = 'Sub'
    op_type = 'Subtract'
    operation = staticmethod(lambda a, b: a - b)


class Mul(Elementwise):
    enabled = False
    op = 'Mul'
    op_type = 'Multiply'
    operation = staticmethod(lambda a, b: a * b)


def both_types_are_integer(a, b):
    return np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer)


class Div(Elementwise):
    enabled = False
    op = 'Div'
    op_type = 'Divide'
    operation = staticmethod(lambda a, b: a // b if both_types_are_integer(a, b) else a / b)


class SquaredDifference(Elementwise):
    enabled = False
    op = 'SquaredDifference'
    op_type = 'SquaredDifference'
    operation = staticmethod(lambda a, b: (a - b) * (a - b))


class Pow(Elementwise):
    enabled = False
    op = 'Pow'
    op_type = 'Power'

    @staticmethod
    def operation(a, b):
        if np.any(b < 0) and np.issubdtype(a.dtype, np.signedinteger):
            return np.array(a.astype(np.float32) ** b, dtype=np.float32)
        return a ** b

    @staticmethod
    def type_infer(node):
        in_type_0 = node.in_port(0).get_data_type()
        in_type_1 = node.in_port(1).get_data_type()
        assert in_type_0 == in_type_1, \
            'Power operation {} has inputs of different data types: {} and {}'.format(
                node.soft_get('name'), in_type_0, in_type_1)
        node.out_port(0).set_data_type(in_type_0)


class LogicalElementwise(Elementwise):
    @staticmethod
    def type_infer(node):
        node.out_port(0).set_data_type(np.bool)


class Greater(LogicalElementwise):
    enabled = False
    op = 'Greater'
    op_type = 'Greater'
    operation = staticmethod(lambda a, b: a > b)


class GreaterEqual(LogicalElementwise):
    enabled = False
    op = 'GreaterEqual'
    op_type = 'GreaterEqual'
    operation = staticmethod(lambda a, b: a >= b)


class Less(LogicalElementwise):
    enabled = False
    op = 'Less'
    op_type = 'Less'
    operation = staticmethod(lambda a, b: a < b)


class LessEqual(LogicalElementwise):
    enabled = False
    op = 'LessEqual'
    op_type = 'LessEqual'
    operation = staticmethod(lambda a, b: a <= b)


class Equal(LogicalElementwise):
    enabled = False
    op = 'Equal'
    op_type = 'Equal'
    operation = staticmethod(lambda a, b: a == b)


class NotEqual(LogicalElementwise):
    enabled = False
    op = 'NotEqual'
    op_type = 'NotEqual'
    operation = staticmethod(lambda a, b: a != b)


class Maximum(Elementwise):
    enabled = False
    op = 'Maximum'
    op_type = 'Maximum'
    operation = staticmethod(lambda a, b: np.maximum(a, b))


class Minimum(Elementwise):
    enabled = False
    op = 'Minimum'
    op_type = 'Minimum'
    operation = staticmethod(lambda a, b: np.minimum(a, b))


class Round(Elementwise):
    enabled = False
    op = 'Round'
    op_type = None
    version = 'extension'
    operation = staticmethod(lambda a: np.round(a))


class LogicalOr(LogicalElementwise):
    enabled = False
    op = 'LogicalOr'
    op_type = 'LogicalOr'
    operation = staticmethod(lambda a, b: np.logical_or(a, b))


class LogicalXor(Elementwise):
    enabled = False
    op = 'LogicalXor'
    op_type = 'LogicalXor'
    operation = staticmethod(lambda a, b: np.logical_xor(a, b))


class LogicalAnd(LogicalElementwise):
    enabled = False
    op = 'LogicalAnd'
    op_type = 'LogicalAnd'
    operation = staticmethod(lambda a, b: np.logical_and(a, b))


class FloorMod(Elementwise):
    enabled = False
    op = 'FloorMod'
    op_type = 'FloorMod'
    operation = staticmethod(lambda a, b: a % b)


class Negative(Elementwise):
    enabled = False
    op = 'Negative'
    op_type = 'Negative'
    operation = staticmethod(lambda a: -a)

    @staticmethod
    def type_infer(node):
        copy_type_infer(node)
