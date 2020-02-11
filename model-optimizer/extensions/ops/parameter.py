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

from mo.graph.graph import Graph
from mo.middle.passes.convert_data_type import np_data_type_to_destination_type
from mo.ops.op import Op, PermuteAttrs


class Parameter(Op):
    op = 'Parameter'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': self.op,
            'type': self.op,

            'infer': self.infer,
            'is_input': True,
            'data_type': None,

            'type_infer': self.type_infer,

            'out_ports_count': 1,
        }
        if 'data_type' not in attrs:
            mandatory_props['data_type'] = np.float32
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def type_infer(node):
        node.out_port(0).set_data_type(node.data_type)

    def supported_attrs(self):
        if self.ir_version == 10:
            return [
                ('shape', lambda node: ','.join([str(i) for i in node.shape])),
                ('element_type', lambda node: np_data_type_to_destination_type(node.data_type)),
            ]
        else:
            return []

    @staticmethod
    def infer(node):
        name = node.soft_get('name', node.id)
        assert node.has_valid('shape'), \
            'Parameter node {} should have `shape` attribute. Please use cli options to set model input shape' \
            ''.format(name)
        node.out_port(0).data.set_shape(node.shape)

        PermuteAttrs.create_permute_attrs(node, attrs=[('shape', 'output:0')])
