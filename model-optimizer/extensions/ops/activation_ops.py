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

import numpy as np

from mo.front.common.partial_infer.eltwise import eltwise_infer
from mo.graph.graph import Graph, Node
from mo.ops.clamp import AttributedClamp
from mo.ops.op import Op

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
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    @classmethod
    def infer(cls, node: Node):
        return eltwise_infer(node, node.operation)


class Sigmoid(Activation):
    op = 'Sigmoid'
    operation = staticmethod(lambda x: 1 / (1 + np.exp(-x)))


class Sin(Activation):
    op = 'Sin'
    operation = staticmethod(lambda x: np.sin(x))


class Sinh(Activation):
    op = 'Sinh'
    operation = staticmethod(lambda x: np.sinh(x))


class Asin(Activation):
    op = 'Asin'
    operation = staticmethod(lambda x: np.arcsin(x))


class Asinh(Activation):
    op = 'Asinh'
    version = 'opset4'
    operation = staticmethod(lambda x: np.arcsinh(x))


class Cos(Activation):
    op = 'Cos'
    operation = staticmethod(lambda x: np.cos(x))


class Cosh(Activation):
    op = 'Cosh'
    operation = staticmethod(lambda x: np.cosh(x))


class Acos(Activation):
    op = 'Acos'
    operation = staticmethod(lambda x: np.arccos(x))


class Acosh(Activation):
    op = 'Acosh'
    version = 'opset4'
    operation = staticmethod(lambda x: np.arccosh(x))


class Tan(Activation):
    op = 'Tan'
    operation = staticmethod(lambda x: np.tan(x))


class Tanh(Activation):
    op = 'Tanh'
    operation = staticmethod(lambda x: np.tanh(x))


class Atan(Activation):
    op = 'Atan'
    operation = staticmethod(lambda x: np.arctan(x))


class Atanh(Activation):
    op = 'Atanh'
    version = 'opset4'
    operation = staticmethod(lambda x: np.arctanh(x))


class ReLU6(AttributedClamp):
    op = 'ReLU6'

    def __init__(self, graph: Graph, attrs: dict):
        relu6_attrs = {'min': 0, 'max': 6}
        relu6_attrs.update(attrs)
        super().__init__(graph, relu6_attrs)


class Exp(Activation):
    op = 'Exp'
    operation = staticmethod(lambda x: np.exp(x))


class ReLU(Activation):
    op = 'ReLU'
    operation = staticmethod(lambda x: np.maximum(0, x))


class Erf(Activation):
    op = 'Erf'
    operation = None


class Floor(Activation):
    op = 'Floor'
    operation = staticmethod(lambda x: np.floor(x))


class Ceiling(Activation):
    op = 'Ceiling'
    operation = staticmethod(lambda x: np.ceil(x))


class Abs(Activation):
    op = 'Abs'
    operation = staticmethod(lambda x: np.abs(x))


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
                values[index] = alpha * (np.exp(x) - 1)
        return values

    @classmethod
    def infer(cls, node: Node):
        return eltwise_infer(node, lambda x, alpha: Elu.elu(x, alpha), alpha=node.alpha)

    def backend_attrs(self):
        return ['alpha']


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
        values = values.astype(float)
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

    operation = staticmethod(lambda x: np.logical_not(x))

    @staticmethod
    def type_infer(node: Node):
        node.out_port(0).set_data_type(np.bool)


class Log(Activation):
    op = 'Log'
    operation = staticmethod(lambda x: np.log(x))


class SoftPlus(Activation):
    op = 'SoftPlus'
    version = 'opset4'
    operation = staticmethod(lambda x: np.ln(np.exp(x) + 1.0))


class Mish(Activation):
    op = 'Mish'
    version = 'opset4'
    operation = staticmethod(lambda x: x * np.tanh(np.ln(np.exp(x) + 1.0)))


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

        input_value = node.in_port(1).data.get_value()
        if input_value is not None and beta is not None:
            node.out_port(0).data.set_value(input_value / (1.0 + np.exp(-input_value * beta)))
