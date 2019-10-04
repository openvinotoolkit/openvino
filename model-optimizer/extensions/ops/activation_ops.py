"""
 Copyright (c) 2019 Intel Corporation

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

from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.front.common.partial_infer.eltwise import eltwise_infer
from mo.graph.graph import Graph
from mo.graph.graph import Node
from mo.ops.clamp import Clamp
from mo.ops.op import Op

activation_ops = ['Sigmoid', 'Tanh', 'ReLU6', 'Exp', 'Elu', 'Not', 'Floor']


class Activation(Op):
    enabled = False
    operation = None
    op = None

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'operation': self.operation,
            'infer': __class__.infer,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    @classmethod
    def infer(cls, node: Node):
        return eltwise_infer(node, node.operation)


class Sigmoid(Activation):
    op = 'Sigmoid'
    operation = staticmethod(lambda x: 1 / (1 + np.exp(-x)))


class Tanh(Activation):
    op = 'Tanh'
    operation = staticmethod(lambda x: np.tanh(x))


class ReLU6(Clamp):
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


class Elu(Activation):
    op = 'Elu'

    def __init__(self, graph: Graph, attrs):
        elu_attrs = {'alpha': 1.0, 'infer': __class__.infer}
        elu_attrs.update(attrs)
        super().__init__(graph, elu_attrs)

    @staticmethod
    def elu(values: np.ndarray, alpha: float):
        values = values.astype(float)
        for index, x in np.ndenumerate(values):
            if x < 0:
                values[index] = alpha * (np.exp(x) - 1)
        return values

    @staticmethod
    def infer(node: Node):
        return eltwise_infer(node, lambda x, alpha: Elu.elu(x, alpha), alpha=node.alpha)

    def backend_attrs(self):
        return ['alpha']


class LeakyReLU(Op):
    op = 'LeakyReLU'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': __class__.op,
            'op': __class__.op,
            'infer': copy_shape_infer,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    def supported_attrs(self):
        return ['negative_slope']


class Not(Activation):
    op = 'Not'
    enabled = False

    operation = staticmethod(lambda x: not x)
