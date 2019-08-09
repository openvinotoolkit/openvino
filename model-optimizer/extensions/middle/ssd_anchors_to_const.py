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

from extensions.middle.AnchorToPriorBox import AnchorToPriorBoxes
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const


class SsdAnchorsMiddleReplacer(MiddleReplacementPattern):
    """
    Replacing subgraph with all anchors constant to constant op with pre calculated prior boxes values.
    """
    enabled = True
    force_clean_up = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'mxnet' and graph.graph['cmd_params'].enable_ssd_gluoncv]

    def run_after(self):
        return [AnchorToPriorBoxes]


    def pattern(self):
        return dict(
            nodes=[
                ('const', dict(op='Const')),
                ('const_data', dict(kind='data')),
                ('reshape0', dict(op='Reshape')),
                ('reshape0_data', dict(kind='data')),
                ('reshape1', dict(op='Reshape')),
                ('reshape1_data', dict(kind='data')),
                ('reshape2', dict(op='Reshape')),
                ('reshape2_data', dict(kind='data')),
                ('reshape3', dict(op='Reshape')),
                ('reshape3_data', dict(kind='data')),
                ('concat', dict(op='Concat')),
            ],
            edges=[
                ('const', 'const_data'),
                ('const_data', 'reshape0'),
                ('reshape0', 'reshape0_data'),
                ('reshape0_data', 'reshape1'),
                ('reshape1', 'reshape1_data'),
                ('reshape1_data', 'reshape2'),
                ('reshape2', 'reshape2_data'),
                ('reshape2_data', 'reshape3'),
                ('reshape3', 'reshape3_data'),
                ('reshape3_data', 'concat'),
        ])

    def replace_pattern(self, graph: Graph, match: dict):
        #self.pattern()['nodes']
        concat_node = match['concat']
        if len(concat_node.out_nodes()[0].out_nodes()) == 0:
            return
        const_values = []
        for in_node_index in concat_node.in_nodes():
            current_node = concat_node.in_port(in_node_index).get_source().node
            for k, v in reversed(self.pattern()['nodes'][:-1]):
                if 'op' in v:
                    assert current_node.op == v['op']
                    current_node = current_node.in_port(0).get_source().node
                    if current_node.op == 'Const':
                        crop_value = current_node.value
                        crop_value = np.reshape(crop_value, (1, -1))
                        const_values.append(crop_value)
                        break
        concat_value = np.concatenate(tuple(const_values), axis=1)
        concat_value = np.reshape(concat_value, (1, 2, -1))
        slice_value = concat_value[0][0]
        for i in range(int(concat_value[0][0].size / 4)):
            index = i * 4
            xmin = slice_value[index] - (slice_value[index+2] / 2)
            ymin = slice_value[index + 1] - (slice_value[index + 3] / 2)
            xmax = slice_value[index] + (slice_value[index + 2] / 2)
            ymax = slice_value[index + 1] + (slice_value[index + 3] / 2)
            slice_value[index] = xmin
            slice_value[index + 1] = ymin
            slice_value[index + 2] = xmax
            slice_value[index + 3] = ymax

        val_node = Const(graph, {'name': concat_node.name + '/const_',
                                 'value': concat_value}).create_node_with_data()
        out_node = concat_node.out_port(0).get_destination().node
        concat_node.out_port(0).disconnect()
        out_node.in_port(2).connect(val_node.in_node().out_port(0))
