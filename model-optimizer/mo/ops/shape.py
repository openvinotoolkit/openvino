"""
 Copyright (C) 2018-2021 Intel Corporation

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
from mo.middle.passes.convert_data_type import np_data_type_to_destination_type
from mo.ops.op import Op
from mo.utils.error import Error


class Shape(Op):
    op = 'ShapeOf'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': 'opset3',

            'output_type': np.int64,
            'infer': self.infer,
            'type_infer': self.type_infer,

            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    def backend_attrs(self):
        version = self.get_opset()
        if version == 'opset3':
            return [
                ('output_type', lambda node: np_data_type_to_destination_type(node.output_type)),
            ]
        elif version == 'opset1':
            return []
        else:
            raise Error('Unknown opset version "{}"'.format(version))

    @staticmethod
    def infer(node):
        name = node.soft_get('name', node.id)
        connected_in_ports = [port for port in node.in_ports().values() if not port.disconnected()]
        assert len(connected_in_ports) == 1, \
            'ShapeOf operation should have exact one input node, but it has {}'.format(len(connected_in_ports))

        input_shape = node.in_port(0).data.get_shape()
        assert input_shape is not None, \
            'Input shape is undefined for ShapeOf node `{}`'.format(node.soft_get('name', node.id))

        assert node.has_valid('output_type'), \
            '`output_type` attribute is not set for ShapeOf node `{}`'.format(name)
        assert node.output_type in [np.int64, np.int32], \
            'ShapeOf `output_type` attribute must be int32 or int64, `{}` found'.format(np.dtype(node.output_type).name)
        if node.has_and_set('stop_value_propagation'):
            node.out_port(0).data.set_shape(input_shape.shape)
        else:
            node.out_port(0).data.set_value(np.array(input_shape, dtype=node.output_type))


    @staticmethod
    def type_infer(node):
        node.out_port(0).set_data_type(node.output_type)
