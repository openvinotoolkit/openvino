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
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const


class AnchorToPriorBoxes(MiddleReplacementPattern):
    """
    Crop anchors consts before replacing subgraph with all anchors
    """
    enabled = True
    force_clean_up = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'mxnet' and graph.graph['cmd_params'].enable_ssd_gluoncv]

    def run_after(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def pattern(self):
        return dict(
            nodes=[
                ('const', dict(op='Const')),
                ('const_data', dict(kind='data')),
                ('slice_like', dict(op='Crop')),
                ('slice_like_out', dict(kind='data')),
                ('reshape', dict(op='Reshape')),
            ],
            edges=[
                ('const', 'const_data'),
                ('const_data', 'slice_like', {'in': 0}),
                ('slice_like', 'slice_like_out'),
                ('slice_like_out', 'reshape'),
            ]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        slice_like = match['slice_like']
        anchor_node = slice_like.in_port(0).get_source().node
        reshape = slice_like.out_nodes()[0].out_node()
        slice_shape = slice_like.out_nodes()[0].shape
        anchor_node.value = np.copy(anchor_node.value[:slice_shape[0], :slice_shape[1],
                                         :slice_shape[2], :slice_shape[3], :slice_shape[4]])
        anchor_node.shape = slice_shape

        val_node = Const(graph, {'name': slice_like.name +'/croped_', 'value': anchor_node.value[:slice_shape[0], :slice_shape[1],
                                         :slice_shape[2], :slice_shape[3], :slice_shape[4]], 'shape': slice_shape}).create_node_with_data()
        slice_like.in_port(0).disconnect()
        slice_like.in_port(1).disconnect()
        slice_like.out_port(0).disconnect()
        reshape.in_port(0).connect(val_node.in_node().out_port(0))
