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

from mo.front.caffe.extractors.utils import get_canonical_axis_index
from mo.graph.graph import Node
from mo.graph.perm_inputs import PermuteInputs
from mo.ops.op import Op


class Squeeze(Op):
    op = 'Squeeze'
    enabled = False

    def __init__(self, graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': self.op,
            'version': 'opset1',

            'reinterp_shape': True,
            'infer': self.infer,

            'in_ports_count': 2,
            'out_ports_count': 1,
        }, attrs)

    @staticmethod
    def infer(node: Node):
        name = node.soft_get('name', node.id)

        connected_input_ports_num = len([port for port in node.in_ports().values() if not port.disconnected()])
        assert connected_input_ports_num in [1, 2], \
            'Squeeze node must have 1 or 2 inputs, but `{}` node has {} inputs'.format(name, connected_input_ports_num)

        input_shape = node.in_port(0).data.get_shape()
        assert input_shape is not None, 'Squeeze node `{}` input shape is unknown'.format(name)
        output_shape = input_shape.copy()

        if node.is_in_port_connected(1):
            raw_axes = node.in_port(1).data.get_value()
            assert raw_axes is not None, 'Squeeze node `{}` 2nd input (axes) value is dynamic. It is not supported'
            axes = [get_canonical_axis_index(input_shape, i) for i in raw_axes]
            assert np.all(input_shape[axes] == 1), \
                'Squeezing non-one dimension is forbidden. Squeeze node `{}` with input_shape={} and axes={}' \
                ''.format(name, input_shape, raw_axes)
        else:
            axes = [i for i, value in enumerate(input_shape) if value == 1]

        output_shape = np.delete(output_shape, axes)
        if np.array_equal(output_shape, []) and node.has_and_set('keep_at_least_1d'):
            # MxNet squeeze keeps at least one dimension for output tensor. Will be resolved on the middle phase
            output_shape = [1]

        if node.in_port(0).data.get_value() is not None:
            node.out_port(0).data.set_value(node.in_port(0).data.get_value().reshape(output_shape))
        else:
            node.out_port(0).data.set_shape(output_shape)

        if node.is_in_port_connected(1):
            PermuteInputs().set_input_permutation(node.in_node(1), node, 'input:0', 'axis')
