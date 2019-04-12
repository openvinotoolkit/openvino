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

import networkx as nx
import numpy as np

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Node
from mo.middle.passes.eliminate import remove_op_nodes
from mo.utils.graph import pseudo_topological_sort


class DumpFakeQuantStat(BackReplacementPattern):
    enabled = True

    def find_and_replace_pattern(self, graph: nx.MultiDiGraph):
        intervals = {}
        for n in pseudo_topological_sort(graph):
            node = Node(graph, n)
            if not node.has('op') or (node.op != 'FakeQuantWithMinMaxVars' and node.op != 'Quantize'):
                continue
            if node.op == 'Quantize':
                # check if input range matches output range
                low_match = np.all(node.in_node(1).value == node.in_node(3).value)
                high_match = np.all(node.in_node(2).value == node.in_node(4).value)
                if not low_match or not high_match:
                    continue

            prev_node = node.in_node().in_node()
            prev_node_id = prev_node.id
            prev_node_out_shape = prev_node.out_node()['shape']
            C = prev_node_out_shape[1]
            assert node.in_node(1).value.size == 1
            assert node.in_node(2).value.size == 1
            min = ', '.join([str(node.in_node(1).value.flatten()[0])] * C)
            max = ', '.join([str(node.in_node(2).value.flatten()[0])] * C)
            intervals[prev_node_id] = {'min': min, 'max': max}
        if intervals:
            if 'statistics' not in graph.graph:
                graph.graph['statistics'] = intervals
            else:
                graph.graph['statistics'].update(intervals)
            remove_op_nodes(graph, {'op': 'FakeQuantWithMinMaxVars'})
            remove_op_nodes(graph, {'op': 'Quantize'})
