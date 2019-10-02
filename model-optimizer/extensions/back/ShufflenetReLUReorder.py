"""
 Copyright (c) 2018-2019 Intel Corporation

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

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph


class ShufflenetReLUReorder(BackReplacementPattern):
    """
    This pass is workaround for GPU plugin
    """
    enabled = False

    def run_before(self):
        from extensions.back.TransposeToPermute import TransposeToPermute
        return [TransposeToPermute]

    def pattern(self):
        return dict(
            nodes=[
                ('relu', dict(kind='op', type='ReLU')),
                ('relu_data', dict(kind='data')),
                ('reshape1', dict(kind='op', type='Reshape')),
                ('reshape1_data', dict(kind='data')),
                ('transpose', dict(kind='op', type='Transpose')),
                ('transpose_data', dict(kind='data')),
                ('reshape2', dict(kind='op', type='Reshape')),
                ('reshape2_data', dict(kind='data')),
                ('conv', dict(kind='op', type='Convolution'))
            ],
            edges=[('relu', 'relu_data'),
                   ('relu_data', 'reshape1'),
                   ('reshape1', 'reshape1_data'),
                   ('reshape1_data', 'transpose'),
                   ('transpose', 'transpose_data'),
                   ('transpose_data', 'reshape2'),
                   ('reshape2', 'reshape2_data'),
                   ('reshape2_data', 'conv'),
                   ]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        relu = match['relu']
        reshape1 = match['reshape1']
        reshape2_data = match['reshape2_data']
        conv = match['conv']

        if np.max(conv.pad) == 0:
            return

        relu_input = relu.in_node()

        # Disconnect InputData-x->ReLU->Data-x->Reshape1
        edge_attrs = graph.get_edge_data(relu.out_node().id, reshape1.id)[0]
        graph.remove_edge(relu_input.id, relu.id)
        graph.remove_edge(relu.out_node().id, reshape1.id)

        # Connect InputData-->Reshape1
        graph.add_edges_from([(relu_input.id, reshape1.id, edge_attrs)])

        # Insert ReLU:  Reshape2Data->ReLU->Data->Convolution
        edge_attrs = graph.get_edge_data(reshape2_data.id, conv.id)[0]
        graph.remove_edge(reshape2_data.id, conv.id)
        graph.add_edges_from([(reshape2_data.id, relu.id, {'in': 0}), (relu.out_node().id, conv.id, edge_attrs)])
