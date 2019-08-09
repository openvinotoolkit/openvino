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

from mo.graph.graph import Graph
from mo.ops.op import Op, PermuteAttrs
from mo.utils.error import Error


class TopK(Op):
    op = 'TopK'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': __class__.op,
            'op': __class__.op,
            'infer': self.infer,
            'axis': None,
            'mode': None,
            'sort': 'none',
            'force_precision_in_ports': {1: 'int32'},
            'in_ports_count': 3,
            'out_ports_count': 2,
        }, attrs)

    def backend_attrs(self):
        return ['axis', 'mode', 'sort']

    @staticmethod
    def infer(node):
        in_ports = node.in_ports()
        connected_ports = [port for port in in_ports.values() if not port.disconnected()]
        assert len(connected_ports) == 2, 'The number of inputs to the TopK layer name "{}" must be equal to 2.' \
                                          ''.format(node.soft_get('name'))

        k = node.in_port(1).data.get_value()
        if k is None:
            raise Error('The value defining number of output elements for layer "{}" is not defined'
                        ''.format(node.soft_get('name')))
        assert node.has_valid('axis'), 'The "axis" attribute is not defined for node {}'.format(node.name)

        input_shape = node.in_port(0).data.get_shape()
        node.axis = len(input_shape) + node.axis if node.axis < 0 else node.axis
        output_shape = input_shape.copy()
        output_shape[node.axis] = k

        PermuteAttrs.create_permute_attrs(node, attrs=[('axis', 'input:0')])

        # setting shape and value if applicable
        if not node.out_port(0).disconnected():
            node.out_port(0).data.set_shape(output_shape)
        if not node.out_port(1).disconnected():
            node.out_port(1).data.set_shape(output_shape)
        if node.in_port(0).data.get_value() is not None:
            # TODO implement value propagation
            pass
