"""
 Copyright (c) 2018 Intel Corporation

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

from mo.graph.graph import erase_node, Node
from mo.middle.passes.eliminate import remove_op_node
from mo.middle.replacement import MiddleReplacementPattern


class ConstSwitchEraser(MiddleReplacementPattern):
    """
    Erases switch operation and its constant input after data and control flow infer
    """
    enabled = True

    def find_and_replace_pattern(self, graph: nx.MultiDiGraph):
        for n in nx.topological_sort(graph):
            if graph.node[n]['kind'] == 'data' or graph.node[n]['op'] != 'Switch':
                continue
            switch_op_node = Node(graph, n)
            pred_id_data_node = switch_op_node.in_node(1)
            erase_node(pred_id_data_node)
            remove_op_node(graph, switch_op_node)
