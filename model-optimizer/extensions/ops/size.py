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
from mo.ops.op import Op


class Size(Op):
    op = 'Size'

    def __init__(self, graph: Graph, attrs: dict):
        assert 'output_type' in attrs, 'Size has mandatory `output_type` attribute'

        mandatory_props = {
            'type': None,
            'op': self.op,

            'output_type': np.int64,
            'infer': self.infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def infer(node: Node):
        name = node.soft_get('name', node.id)
        connected_in_ports = [port for port in node.in_ports().values() if not port.disconnected()]
        assert len(connected_in_ports) == 1, \
            'Size operation should have exact one input node, but it has {}'.format(len(connected_in_ports))

        input_shape = node.in_port(0).data.get_shape()
        assert input_shape is not None, \
            'Input shape is undefined for Size node `{}`'.format(node.soft_get('name', node.id))

        assert node.has_valid('output_type'), \
            '`output_type` attribute is not set for Size node `{}`'.format(name)
        assert node.output_type in [np.int64, np.int32], \
            'Size `output_type` attribute must be int32 or int64, `{}` found'.format(np.dtype(node.output_type).name)

        node.out_port(0).data.set_value(np.array(np.prod(input_shape), dtype=node.output_type))
