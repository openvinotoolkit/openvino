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

from mo.graph.graph import Node, Graph
from mo.graph.perm_inputs import PermuteInputs
from mo.ops.op import Op


class Reshape(Op):
    op = 'Reshape'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': self.op,

            'infer': self.infer,

            'special_zero': True,
            'reinterp_shape': True,

            'in_ports_count': 2,
            'out_ports_count': 1,
        }, attrs)

    def supported_attrs(self):
        return ['special_zero']

    @staticmethod
    def infer(node: Node):
        name = node.soft_get('name', node.id)

        connected_inputs = {idx: port for idx, port in node.in_ports().items() if not port.disconnected()}
        assert len(connected_inputs) == 2 and all([i in connected_inputs for i in range(2)]), \
            "Reshape should have 2 connected input ports, but it doesn't for node: `{}`. Ports: {}" \
            "".format(name, connected_inputs)

        input_shape = node.in_port(0).data.get_shape()
        assert input_shape is not None

        new_shape = node.in_port(1).data.get_value()
        assert new_shape is not None, 'Dynamic Reshape second input is not supported. Node {}'.format(name)

        assert np.argwhere(new_shape == -1).size <= 1, \
            'Reshape second input should not have several `-1` values set. ' \
            'Node: {}, reshape second input value {}'.format(name, new_shape)

        num_of_input_elements = np.prod(input_shape)
        num_of_output_elements = 1
        for index, x in enumerate(new_shape):
            if x == 0 and node.has_and_set('special_zero'):
                num_of_output_elements *= input_shape[index]
            elif x != -1:
                num_of_output_elements *= x

        undefined_dim = num_of_input_elements // num_of_output_elements
        output_shape = []
        for index, x in enumerate(new_shape):
            if x == 0 and node.has_and_set('special_zero'):
                output_shape.append(input_shape[index])
            elif x == -1:
                output_shape.append(undefined_dim)
            else:
                output_shape.append(x)

        assert np.prod(input_shape) == np.prod(output_shape), \
            "Number of elements in input {} and output {} of reshape node {} mismatch" \
            "".format(input_shape, output_shape, name)

        PermuteInputs().set_input_permutation(node.in_node(1), node, 'output:0', 'shape')

        if node.in_port(0).data.get_value() is not None:
            node.out_port(0).data.set_value(node.in_port(0).data.get_value().reshape(output_shape))
        else:
            node.out_port(0).data.set_shape(output_shape)
