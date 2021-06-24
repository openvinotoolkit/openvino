# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from mo.front.common.partial_infer.eltwise import eltwise_infer, bias_add_infer
from mo.graph.graph import Graph, Node
from mo.middle.passes.infer import copy_type_infer
from mo.ops.op import Op
from mo.pipeline.common import convert_const_node_value_type
from mo.utils.error import Error


def override_data_type_of_constant(node: Node):
    in_type_0 = node.in_port(0).get_data_type()
    in_type_1 = node.in_port(1).get_data_type()
    if in_type_0 != in_type_1:
        # in case of input values data type mismatch we try to change the type of the constant to match the type of
        # another input. The input values data type mismatch occur when the MO performs replacement of some
        # operations like SquaredDifference of inputs with floating point data type to Power layer with the integer
        # power value 2, or when replacing Neg operation with Mul with -1 as second input.
        in_node_0 = node.in_port(0).get_source().node
        in_node_1 = node.in_port(1).get_source().node

        if in_node_0.op != 'Const' and in_node_1.op != 'Const':
            raise Error("Elementwise operation '{}' has inputs of different data types: '{}' and '{}' "
                        "that cannot be aligned".format(node.soft_get('name'), in_type_0, in_type_1))

        if in_node_0.op == 'Const':
            node_to_convert, src_type, dst_type = in_node_0, in_type_0, in_type_1
        else:
            node_to_convert, src_type, dst_type = in_node_1, in_type_1, in_type_0
        log.error("Changing Const node '{}' data type from {} to {} for Elementwise operation".format(
            node_to_convert.soft_get('name', node_to_convert.id), src_type, dst_type),
            extra={'is_warning': True})
        convert_const_node_value_type(node_to_convert, dst_type)


class Elementwise(Op):
    enabled = False
    operation = None
    op = None
    op_type = None
    version = 'opset1'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': self.op_type,
            'version': self.version,
            'infer': lambda node: eltwise_infer(node, self.operation),
            'type_infer': self.type_infer,
            'can_be_bias': True,
            'can_be_fused': True,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'is_eltwise': True,
            'stop_value_propagation': False,
            'auto_broadcast': 'numpy'
        }, attrs)

    @staticmethod
    def type_infer(node):
        override_data_type_of_constant(node)
        node.out_port(0).set_data_type(node.in_port(0).get_data_type())

    def backend_attrs(self):
        return ['auto_broadcast']


class UnaryElementwise(Elementwise):
    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {**{
            'in_ports_count': 1,
        }, **attrs})

    @staticmethod
    def type_infer(node):
        copy_type_infer(node)

    def backend_attrs(self):
        return []


class Add(Elementwise):
    op = 'Add'
    op_type = 'Add'
    operation = staticmethod(lambda a, b: a + b)


class BiasAdd(Add):
    op_type = 'BiasAdd'

    def __init__(self, graph: Graph, attrs: dict):
        attrs.update({'infer': lambda node: bias_add_infer(node, self.operation)})
        super().__init__(graph, attrs)


class Sub(Elementwise):
    op = 'Sub'
    op_type = 'Subtract'
    operation = staticmethod(lambda a, b: a - b)


class Mul(Elementwise):
    op = 'Mul'
    op_type = 'Multiply'
    operation = staticmethod(lambda a, b: a * b)


def both_types_are_integer(a, b):
    return np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer)


class Div(Elementwise):
    op = 'Div'
    op_type = 'Divide'
    operation = staticmethod(lambda a, b: a // b if both_types_are_integer(a, b) else a / b)


class SquaredDifference(Elementwise):
    op = 'SquaredDifference'
    op_type = 'SquaredDifference'
    operation = staticmethod(lambda a, b: (a - b) * (a - b))


class Pow(Elementwise):
    op = 'Pow'
    op_type = 'Power'

    @staticmethod
    def operation(a, b):
        if np.any(b < 0) and np.issubdtype(a.dtype, np.signedinteger):
            return np.array(a.astype(np.float32) ** b, dtype=np.float32)
        return a ** b


class LogicalElementwise(Elementwise):
    @staticmethod
    def type_infer(node):
        override_data_type_of_constant(node)
        node.out_port(0).set_data_type(np.bool)


class Greater(LogicalElementwise):
    op = 'Greater'
    op_type = 'Greater'
    operation = staticmethod(lambda a, b: a > b)


class GreaterEqual(LogicalElementwise):
    op = 'GreaterEqual'
    op_type = 'GreaterEqual'
    operation = staticmethod(lambda a, b: a >= b)


class Less(LogicalElementwise):
    op = 'Less'
    op_type = 'Less'
    operation = staticmethod(lambda a, b: a < b)


class LessEqual(LogicalElementwise):
    op = 'LessEqual'
    op_type = 'LessEqual'
    operation = staticmethod(lambda a, b: a <= b)


class Equal(LogicalElementwise):
    op = 'Equal'
    op_type = 'Equal'
    operation = staticmethod(lambda a, b: a == b)


class NotEqual(LogicalElementwise):
    op = 'NotEqual'
    op_type = 'NotEqual'
    operation = staticmethod(lambda a, b: a != b)


class Maximum(Elementwise):
    op = 'Maximum'
    op_type = 'Maximum'
    operation = staticmethod(lambda a, b: np.maximum(a, b))


class Minimum(Elementwise):
    op = 'Minimum'
    op_type = 'Minimum'
    operation = staticmethod(lambda a, b: np.minimum(a, b))


class Round(UnaryElementwise):
    op = 'Round'
    op_type = 'Round'
    version = 'opset5'

    def __init__(self, graph: Graph, attrs):
        round_attrs = {'mode': 'half_to_even',
                       'infer': self.infer
                       }
        round_attrs.update(attrs)
        super().__init__(graph, round_attrs)

    def backend_attrs(self):
        return ['mode']

    @classmethod
    def infer(cls, node: Node):
        node.out_port(0).data.set_shape(node.in_port(0).data.get_shape())

        a = node.in_port(0).data.get_value()
        if a is not None:
            assert node.soft_get('mode') in ['half_to_even', 'half_away_from_zero'], \
                'Round node {} has unsupported "mode" attribute value: {}'.format(node.soft_get('name', node.id),
                                                                                  node.soft_get('mode'))
            if node.mode == 'half_away_from_zero':
                mask = (a >= 0)
                out = np.empty_like(a)
                out[mask] = np.floor(a[mask] + 0.5)
                out[~mask] = np.ceil(a[~mask] - 0.5)
            else:
                out = np.round(a)
            node.out_port(0).data.set_value(out)


class LogicalOr(LogicalElementwise):
    op = 'LogicalOr'
    op_type = 'LogicalOr'
    operation = staticmethod(lambda a, b: np.logical_or(a, b))


class LogicalXor(Elementwise):
    op = 'LogicalXor'
    op_type = 'LogicalXor'
    operation = staticmethod(lambda a, b: np.logical_xor(a, b))


class LogicalAnd(LogicalElementwise):
    op = 'LogicalAnd'
    op_type = 'LogicalAnd'
    operation = staticmethod(lambda a, b: np.logical_and(a, b))


class FloorMod(Elementwise):
    op = 'FloorMod'
    op_type = 'FloorMod'
    operation = staticmethod(lambda a, b: a % b)


class Negative(UnaryElementwise):
    op = 'Negative'
    op_type = 'Negative'
    operation = staticmethod(lambda a: -a)
