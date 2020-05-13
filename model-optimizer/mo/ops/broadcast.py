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

from mo.graph.graph import Node, Graph
from mo.graph.perm_inputs import PermuteInputs
from mo.ops.op import Op
from mo.utils.broadcasting import bi_directional_shape_broadcasting, uni_directional_shape_broadcasting, \
    uni_directional_broadcasting, bi_directional_broadcasting
from mo.utils.error import Error


class Broadcast(Op):
    """ Broadcast tensor to a given shape with optional axis parameter

        Inputs:
            [0] - tensor to be broadcasted
            [1] - shape to be broadcast to
            [2] - optional axis parameter that which axis are allowed to be broadcasted
    """

    op = 'Broadcast'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': __class__.op,
            'op': __class__.op,
            'version': 'opset3',
            'mode': 'numpy',
            'in_ports_count': 3,
            'out_ports_count': 1,
            'force_precision_in_ports':
                {1: 'int64' if graph.graph['cmd_params'].generate_experimental_IR_V10 else 'int32',
                 2: 'int64' if graph.graph['cmd_params'].generate_experimental_IR_V10 else 'int32',
                 },
            'infer': __class__.infer,
        }, attrs)

    def supported_attrs(self):
        return ['mode']

    @staticmethod
    def infer(node: Node):
        node_name = node.soft_get('name', node.id)

        input_shape = node.in_port(0).data.get_shape()
        input_value = node.in_port(0).data.get_value()
        target_shape = node.in_port(1).data.get_value()
        assert target_shape is not None, 'Output shape is not defined for node "{}"'.format(node_name)
        assert node.has_and_set('mode'), 'Broadcasting mode is not defined for node "{}"'.format(node_name)

        if node.mode == 'numpy':
            node.out_port(0).data.set_shape(uni_directional_shape_broadcasting(input_shape, target_shape))
        elif node.mode == 'bidirectional':
            node.out_port(0).data.set_shape(bi_directional_shape_broadcasting(input_shape, target_shape))
        else:
            raise Error('The node "{}" has unsupported mode "{}"'.format(node_name, node.mode))

        PermuteInputs().set_input_permutation(node.in_node(1), node, 'output:0', 'shape')

        if input_value is not None and not node.has_and_set('stop_value_propagation'):
            if node.mode == 'numpy':
                node.out_port(0).data.set_value(uni_directional_broadcasting(input_value, target_shape))
            elif node.mode == 'bidirectional':
                node.out_port(0).data.set_value(bi_directional_broadcasting(input_value, target_shape))
