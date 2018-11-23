"""
 Copyright (c) 2018 Intel Corporation

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

import networkx as nx
import numpy as np

from mo.front.common.partial_infer.eltwise import eltwise_infer
from mo.graph.graph import Node
from mo.ops.op import Op


class Activation(Op):
    op = 'Activation'

    @staticmethod
    def elu(values, alpha):
        values = values.astype(float)
        for index, x in np.ndenumerate(values):
            if x < 0:
                values[index] = alpha * (np.exp(x) - 1)
        return values

    operations = {
        'tanh': lambda x: np.tanh(x),
        'elu': lambda x, alpha: Activation.elu(x, alpha),
        'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
        'relu6': lambda x: np.maximum(0, np.minimum(x, 6))
    }

    def __init__(self, graph: nx.MultiDiGraph, attrs: dict):
        super().__init__(graph, {
            'type': __class__.op,
            'op': __class__.op,
            'infer': Activation.infer
        }, attrs)

    @classmethod
    def infer(cls, node: Node):
        if node.operation == 'elu':
            # Set default value for alpha in case when it is not specified
            node['alpha'] = node.alpha if node.has_valid('alpha') else 1.0
            return eltwise_infer(node, cls.operations[node.operation], alpha=node.alpha)
        return eltwise_infer(node, cls.operations[node.operation])

    def supported_attrs(self):
        return ['operation']

    def backend_attrs(self):
        return [('type', 'operation'), 'alpha']  # operation --> type
