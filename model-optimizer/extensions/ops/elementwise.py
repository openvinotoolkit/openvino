"""
 Copyright (c) 2018-2019 Intel Corporation

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

import numpy as np

from mo.front.common.partial_infer.eltwise import eltwise_infer, bias_add_infer
from mo.graph.graph import Graph
from mo.middle.passes.convert_data_type import data_type_str_to_np
from mo.ops.op import Op
from mo.utils.error import Error


class Elementwise(Op):
    enabled = False
    operation = None
    op = None
    op_type = None

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': self.op_type,
            'infer': lambda node: eltwise_infer(node, self.operation),
            'type_infer': self.type_infer,
            'can_be_bias': True,
            'can_be_fused': True,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'is_eltwise': True,
        }, attrs)

    @staticmethod
    def type_infer(node):
        in_type_0 = node.in_port(0).get_data_type()
        in_type_1 = node.in_port(1).get_data_type()
        if in_type_0 != in_type_1:
            raise Error('Elementwise operation {} has inputs of different data types: {} and {}'.format(
                        node.soft_get('name'), in_type_0, in_type_1))
        node.out_port(0).set_data_type(in_type_0)


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


class Div(Elementwise):
    enabled = False
    op = 'Div'
    op_type = 'Divide'
    operation = staticmethod(lambda a, b: a / b)


class Pow(Elementwise):
    enabled = False
    op = 'Pow'
    op_type = 'Pow'

    @staticmethod
    def operation(a, b):
        if np.any(b < 0) and np.issubdtype(a.dtype, np.signedinteger):
            return np.array(a.astype(np.float32) ** b, dtype=np.float32)
        return a ** b

    @staticmethod
    def type_infer(node):
        # dynamic power output data type is complicate to predict, so we set float data type by default,
        # if we haven't got actual value
        value = node.out_port(0).data.get_value()
        if value is not None:
            node.out_port(0).set_data_type(value.dtype)
        else:
            node.out_port(0).set_data_type(data_type_str_to_np(node.graph.graph['cmd_params'].data_type))


class Greater(Elementwise):
    enabled = False
    op = 'Greater'
    op_type = 'Greater'
    operation = staticmethod(lambda a, b: a > b)


class GreaterEqual(Elementwise):
    enabled = False
    op = 'GreaterEqual'
    op_type = 'GreaterEqual'
    operation = staticmethod(lambda a, b: a >= b)


class Less(Elementwise):
    enabled = False
    op = 'Less'
    op_type = 'Less'
    operation = staticmethod(lambda a, b: a < b)


class LessEqual(Elementwise):
    enabled = False
    op = 'LessEqual'
    op_type = 'LessEqual'
    operation = staticmethod(lambda a, b: a <= b)


class Equal(Elementwise):
    enabled = False
    op = 'Equal'
    op_type = 'Equal'
    operation = staticmethod(lambda a, b: a == b)


class NotEqual(Elementwise):
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
    operation = staticmethod(lambda a: np.round(a))


class LogicalOr(Elementwise):
    enabled = False
    op = 'LogicalOr'
    op_type = 'LogicalOr'
    operation = staticmethod(lambda a, b: bool(a) or bool(b))


class LogicalAnd(Elementwise):
    enabled = False
    op = 'LogicalAnd'
    op_type = 'LogicalAnd'
    operation = staticmethod(lambda a, b: bool(a) and bool(b))
