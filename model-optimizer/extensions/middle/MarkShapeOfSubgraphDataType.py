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
from mo.graph.graph import Graph, Node
from mo.middle.replacement import MiddleReplacementPattern
from mo.utils.graph import bfs_search_apply_on_shapeof_subgraph_nodes


class MarkShapeOfNodes(MiddleReplacementPattern):
    enabled = True

    def run_after(self):
        from extensions.middle.pass_separator import PostMiddleStart
        return [PostMiddleStart]

    def run_before(self):
        from extensions.middle.MarkSubgraphsWithCorrectLayout import MarkSubGraphsWithCorrectLayout
        return [MarkSubGraphsWithCorrectLayout]

    @staticmethod
    def mark_node(node: Node):
        node['in_shape_subgraph'] = True
        for out_port in node.out_ports().values():
            node.out_node(out_port.idx)['correct_data_type'] = True
            node.out_node(out_port.idx)['in_shape_subgraph'] = True

    def find_and_replace_pattern(self, graph: Graph):
        bfs_search_apply_on_shapeof_subgraph_nodes(graph.get_op_nodes(op='ShapeOf'), action=MarkShapeOfNodes.mark_node)
