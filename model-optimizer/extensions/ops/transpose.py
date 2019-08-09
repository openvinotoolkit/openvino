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

from mo.graph.graph import Graph
from mo.graph.perm_inputs import PermuteInputs
from mo.ops.op import Op


class Transpose(Op):
    op = 'Transpose'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': __class__.op,
            'op': __class__.op,
            'infer': self.infer,
            'force_precision_in_ports': {1: 'int64'},
            'in_ports_count': 2,
            'out_ports_count': 1,
        }, attrs)

    @staticmethod
    def infer(node):
        # order parameter calculation and checks
        in_ports = node.in_ports()
        connected_ports = [port for port in in_ports.values() if not port.disconnected()]
        input_shape = node.in_port(0).data.get_shape()

        if node.has_and_set('reverse_order'):
            assert len(connected_ports) == 1 and 0 in in_ports, \
                'Cannot infer `{}` due to both order and reverse_order was set'.format(node.soft_get('name'))
            order = np.arange(len(input_shape))[::-1]  # Reverse order
        else:
            assert len(connected_ports) == 2 and 0 in in_ports and 1 in in_ports, \
                "{} node `{}` should have 2 input ports, where 0-input is a data input and 1-input represents " \
                "Transpose `order`".format(node.op, node.id)
            order = node.in_port(1).data.get_value()
            assert order is not None, 'Cannot infer `{}` because order is None'.format(node.soft_get('name'))
            PermuteInputs().set_input_permutation(node.in_node(1), node, 'input:0', 'order')

        # setting shape and value if applicable
        node.out_port(0).data.set_shape(input_shape[order])
        if node.in_port(0).data.get_value() is not None:
            node.out_port(0).data.set_value(np.transpose(node.in_port(0).data.get_value(), axes=order))

