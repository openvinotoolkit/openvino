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

from mo.graph.graph import Graph, Node
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.op import Op


class SharedWeightsDuplication(MiddleReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_after(self):
        from extensions.middle.CheckForCycle import CheckForCycle
        return [CheckForCycle]

    def run_before(self):
        from extensions.middle.pass_separator import PreMiddleStart
        return [PreMiddleStart]

    def find_and_replace_pattern(self, graph: Graph):
        """
        This function finds all const data nodes that have more that one consumer and then duplicate them
        """
        data_nodes = [Node(graph, id) for id in graph.nodes() if Node(graph, id).soft_get('kind') == 'data']
        for node in data_nodes:
            # Check that node has const values and more than one consumer
            if len(node.in_nodes()) and node.in_node().soft_get('type') == 'Const' and len(node.out_nodes()) > 1 and \
                            node.value is not None:
                # Here we delete all edges between base node and it's consumers (except first), and then duplicate this
                # node to connect with other consumers
                for v, d in node.get_outputs():
                    out_node = Node(graph, v)
                    e_attrs = d
                    graph.remove_edge(node.id, out_node.id)
                    data = Op.create_input_data_node(graph, "Copy_{}".format(node.id), np.array(node.value),
                                                     graph.node[node.id])

                    graph.add_edges_from([(data.id, out_node.id, e_attrs)])

