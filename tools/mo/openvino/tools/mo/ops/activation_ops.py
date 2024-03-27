# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.eltwise import eltwise_infer
from openvino.tools.mo.front.common.partial_infer.utils import reverse_bypass_infer
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.ops.clamp import AttributedClamp
from openvino.tools.mo.ops.op import Op

activation_ops = ['Sigmoid', 'Tanh', 'ReLU6', 'Exp', 'Elu', 'LogicalNot', 'Floor', 'Ceiling']


class Activation(Op):
    enabled = False
    operation = None
    op = None
    version = 'opset1'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'operation': self.operation,
            'version': self.version,
            'infer': self.infer,
            'reverse_infer': lambda node: reverse_bypass_infer(node, in_ports=[0]),
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    @classmethod
    def infer(cls, node: Node):
        return eltwise_infer(node, node.operation)


class Sigmoid(Activation):
    op = 'Sigmoid'
    operation = staticmethod(lambda x: 1 / (1 + np.ma.exp(-x)))


class Sin(Activation):
    op = 'Sin'
    operation = staticmethod(lambda x: np.ma.sin(x))


class Sinh(Activation):
    op = 'Sinh'
    operation = staticmethod(lambda x: np.ma.sinh(x))


class Asin(Activation):
    op = 'Asin'
    operation = staticmethod(lambda x: np.ma.arcsin(x))


class Asinh(Activation):
    op = 'Asinh'
    version = 'opset4'
    operation = staticmethod(lambda x: np.arcsinh(x))


class Cos(Activation):
    op = 'Cos'
    operation = staticmethod(lambda x: np.ma.cos(x))


class Cosh(Activation):
    op = 'Cosh'
    operation = staticmethod(lambda x: np.ma.cosh(x))


class Acos(Activation):
    op = 'Acos'
    operation = staticmethod(lambda x: np.ma.arccos(x))


class Acosh(Activation):
    op = 'Acosh'
    version = 'opset4'
    operation = staticmethod(lambda x: np.ma.arccosh(x))


class Tan(Activation):
    op = 'Tan'
    operation = staticmethod(lambda x: np.ma.tan(x))


class Tanh(Activation):
    op = 'Tanh'
    operation = staticmethod(lambda x: np.ma.tanh(x))


class Atan(Activation):
    op = 'Atan'
    operation = staticmethod(lambda x: np.ma.arctan(x))


class Atanh(Activation):
    op = 'Atanh'
    version = 'opset4'
    operation = staticmethod(lambda x: np.ma.arctanh(x))


class ReLU6(AttributedClamp):
    def __init__(self, graph: Graph, attrs: dict):
        relu6_attrs = {'min': 0, 'max': 6}
        relu6_attrs.update(attrs)
        super().__init__(graph, relu6_attrs)


class Exp(Activation):
    op = 'Exp'
    operation = staticmethod(lambda x: np.ma.exp(x))


class ReLU(Activation):
    op = 'ReLU'
    operation = staticmethod(lambda x: np.ma.maximum(0, x))


class Erf(Activation):
    op = 'Erf'
    operation = None


class Floor(Activation):
    op = 'Floor'
    operation = staticmethod(lambda x: x if np.issubdtype(x.dtype, np.integer) else np.ma.floor(x))


class Ceiling(Activation):
    op = 'Ceiling'
    operation = staticmethod(lambda x: np.ma.ceil(x))


class Abs(Activation):
    op = 'Abs'
    operation = staticmethod(lambda x: np.ma.abs(x))


class Sign(Activation):
    op = 'Sign'
    operation = staticmethod(lambda x: np.sign(x))


class Elu(Activation):
    op = 'Elu'

    def __init__(self, graph: Graph, attrs):
        elu_attrs = {'alpha': 1.0}
        elu_attrs.update(attrs)
        super().__init__(graph, elu_attrs)

    @staticmethod
    def elu(values: np.ndarray, alpha: float):
        values = values.astype(float)
        for index, x in np.ndenumerate(values):
            if x < 0:
                values[index] = alpha * (np.ma.exp(x) - 1)
        return values

    @classmethod
    def infer(cls, node: Node):
        return eltwise_infer(node, lambda x, alpha: Elu.elu(x, alpha), alpha=node.alpha)

    def backend_attrs(self):
        return ['alpha']


class ThresholdedRelu(Activation):
    # The operation will be decomposed to primitive operations
    op = 'ThresholdedRelu'

    def __init__(self, graph: Graph, attrs):
        trelu_attrs = {'alpha': 1.0, 'type': None}
        trelu_attrs.update(attrs)
        super().__init__(graph, trelu_attrs)

    @staticmethod
    def thresholded_relu(values: np.ndarray, alpha: float):
        values = values.astype(float)
        for index, x in np.ndenumerate(values):
            values[index] = values[index] * (x > alpha)
        return values

    @classmethod
    def infer(cls, node: Node):
        return eltwise_infer(node, lambda x, alpha: ThresholdedRelu.thresholded_relu(x, alpha), alpha=node.alpha)


class LeakyReLU(Op):
    op = 'LeakyReLU'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'infer': self.infer,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    @staticmethod
    def leaky_relu(values: np.ndarray, negative_slope: float):
        for index, x in np.ndenumerate(values):
            if x < 0:
                values[index] = negative_slope * x
        return values

    @staticmethod
    def infer(node: Node):
        return eltwise_infer(node, lambda x, negative_slope: LeakyReLU.leaky_relu(x, negative_slope),
                             negative_slope=node.negative_slope)

    def supported_attrs(self):
        return ['negative_slope']


class LogicalNot(Activation):
    op = 'LogicalNot'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        not_attrs = {'type_infer': self.type_infer}
        not_attrs.update(attrs)
        super().__init__(graph, not_attrs)

    operation = staticmethod(lambda x: np.ma.logical_not(x))

    @staticmethod
    def type_infer(node: Node):
        node.out_port(0).set_data_type(bool)


class Log(Activation):
    op = 'Log'
    operation = staticmethod(lambda x: np.ma.log(x))


class SoftPlus(Activation):
    op = 'SoftPlus'
    version = 'opset4'
    operation = staticmethod(lambda x: np.ma.log(np.ma.exp(x) + 1.0))


class Mish(Activation):
    op = 'Mish'
    version = 'opset4'
    operation = staticmethod(lambda x: x * np.ma.tanh(np.ma.log(np.ma.exp(x) + 1.0)))


class HSwish(Activation):
    op = 'HSwish'
    version = 'opset4'
    operation = staticmethod(lambda x: x * np.ma.minimum(np.ma.maximum(x + 3.0, 0.0), 6.0) / 6.0)


class HSigmoid(Activation):
    op = 'HSigmoid'
    version = 'opset5'
    operation = staticmethod(lambda x: np.ma.minimum(np.ma.maximum(x + 3.0, 0.0), 6.0) / 6.0)


class Swish(Op):
    op = 'Swish'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': self.op,
            'type': self.op,
            'version': 'opset4',

            'infer': self.infer,

            'in_ports_count': 2,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def infer(node: Node):
        node_name = node.soft_get('name', node.id)
        node.out_port(0).data.set_shape(node.in_port(0).data.get_shape())

        beta = 1.0
        if node.is_in_port_connected(1):
            beta = node.in_port(1).data.get_value()
            if beta is not None:
                assert beta.ndim == 0, 'The "beta" value for node {} must be a scalar'.format(node_name)
                beta = beta.item()

        input_value = node.in_port(0).data.get_value()
        if input_value is not None and beta is not None:
            node.out_port(0).data.set_value(input_value / (1.0 + np.exp(-input_value * beta)))


class SoftSign(Activation):
    op = "SoftSign"
    version = "opset9"
    operation = staticmethod(lambda x: x / (np.ma.abs(x) + 1))
