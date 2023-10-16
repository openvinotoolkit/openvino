# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.eltwise import eltwise_infer, bias_add_infer, eltwise_reverse_infer
from openvino.tools.mo.front.common.partial_infer.utils import float32_array, reverse_bypass_infer
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.middle.passes.infer import copy_type_infer
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.utils.type_utils import override_data_type_of_constant


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
            'reverse_infer': eltwise_reverse_infer,
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
            'reverse_infer': lambda node: reverse_bypass_infer(node, in_ports=[0]),
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
            return float32_array(a.astype(np.float32) ** b)
        return a ** b


class LogicalElementwise(Elementwise):
    @staticmethod
    def type_infer(node):
        override_data_type_of_constant(node)
        node.out_port(0).set_data_type(bool)


class Greater(LogicalElementwise):
    op = 'Greater'
    op_type = 'Greater'
    operation = staticmethod(lambda a, b: np.ma.greater(a, b))


class GreaterEqual(LogicalElementwise):
    op = 'GreaterEqual'
    op_type = 'GreaterEqual'
    operation = staticmethod(lambda a, b: np.ma.greater_equal(a, b))


class Less(LogicalElementwise):
    op = 'Less'
    op_type = 'Less'
    operation = staticmethod(lambda a, b: np.ma.less(a, b))


class LessEqual(LogicalElementwise):
    op = 'LessEqual'
    op_type = 'LessEqual'
    operation = staticmethod(lambda a, b: np.ma.less_equal(a, b))


class Equal(LogicalElementwise):
    op = 'Equal'
    op_type = 'Equal'
    operation = staticmethod(lambda a, b: np.ma.equal(a, b))


class NotEqual(LogicalElementwise):
    op = 'NotEqual'
    op_type = 'NotEqual'
    operation = staticmethod(lambda a, b: np.ma.not_equal(a, b))


class Maximum(Elementwise):
    op = 'Maximum'
    op_type = 'Maximum'
    operation = staticmethod(lambda a, b: np.ma.maximum(a, b))


class Minimum(Elementwise):
    op = 'Minimum'
    op_type = 'Minimum'
    operation = staticmethod(lambda a, b: np.ma.minimum(a, b))


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
                out = np.ma.empty_like(a)
                out[mask] = np.ma.floor(a[mask] + 0.5)
                out[~mask] = np.ma.ceil(a[~mask] - 0.5)
            else:
                out = np.ma.round(a)
            node.out_port(0).data.set_value(out)


class LogicalOr(LogicalElementwise):
    op = 'LogicalOr'
    op_type = 'LogicalOr'
    operation = staticmethod(lambda a, b: np.ma.logical_or(a, b))


class LogicalXor(Elementwise):
    op = 'LogicalXor'
    op_type = 'LogicalXor'
    operation = staticmethod(lambda a, b: np.ma.logical_xor(a, b))


class LogicalAnd(LogicalElementwise):
    op = 'LogicalAnd'
    op_type = 'LogicalAnd'
    operation = staticmethod(lambda a, b: np.ma.logical_and(a, b))


class FloorMod(Elementwise):
    op = 'FloorMod'
    op_type = 'FloorMod'
    operation = staticmethod(lambda a, b: np.ma.fmod(a, b))


class Mod(Elementwise):
    op = 'Mod'
    op_type = 'Mod'
    operation = staticmethod(lambda a, b: np.ma.mod(a, b))


class Negative(UnaryElementwise):
    op = 'Negative'
    op_type = 'Negative'
    operation = staticmethod(lambda a: -a)


class Sqrt(UnaryElementwise):
    op = 'Sqrt'
    op_type = 'Sqrt'

    @staticmethod
    def operation(a):
        if np.issubdtype(a.dtype, np.signedinteger):
            return float32_array(a.astype(np.float32) ** 0.5)
        return a ** 0.5


class BitwiseAnd(Elementwise):
    op = 'BitwiseAnd'
    op_type = 'BitwiseAnd'
    version = 'opset13'


class BitwiseOr(Elementwise):
    op = 'BitwiseOr'
    op_type = 'BitwiseOr'
    version = 'opset13'


class BitwiseXor(Elementwise):
    op = 'BitwiseXor'
    op_type = 'BitwiseXor'
    version = 'opset13'


class BitwiseNot(UnaryElementwise):
    op = 'BitwiseNot'
    op_type = 'BitwiseNot'
    version = 'opset13'
