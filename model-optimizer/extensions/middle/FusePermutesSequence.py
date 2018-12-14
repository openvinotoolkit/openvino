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
import numpy as np

from extensions.middle.ConvertLayoutDependentOperations import ConvertLayoutDependentOperations
from mo.graph.graph import Node
from mo.middle.passes.eliminate import merge_data_nodes, graph_clean_up_tf
from mo.middle.passes.fusing.helpers import get_next_operation
from mo.middle.replacement import MiddleReplacementPattern
from mo.utils.error import Error


class FusePermutesSequence(MiddleReplacementPattern):
    """
         This pass finds sequence of Permute operations and merge them to single Permute operation
         In case if resulting Permutation do nothing, we just remove it
    """

    enabled = True

    def run_after(self):
        return [ConvertLayoutDependentOperations]

    def find_and_replace_pattern(self, graph: nx.MultiDiGraph):
        for node in list(graph.nodes()):
            if node not in graph.nodes():
                continue
            permute_node = Node(graph, node)
            if permute_node.has_valid('type') and permute_node.type == 'Permute':
                list_of_permutes = [permute_node]
                # Get sequence of permutations
                node = permute_node
                while True:
                    next_ops = get_next_operation(node)
                    if len(next_ops) != 1:
                        break

                    next_op = next_ops[0]
                    if next_op.has_valid('type') and next_op.type == 'Permute':
                        list_of_permutes.append(next_op)
                        node = next_op
                    else:
                        break

                final_permutation = np.array([x for x in range(len(list_of_permutes[0].order))], dtype=np.int64)
                for permute in list_of_permutes:
                    if not permute.has_valid('order'):
                        raise Error("Permute node {} has wrong attribute order = None".format(permute.name))
                    final_permutation = final_permutation[np.array(permute.order, dtype=np.int64)]

                if np.array_equal(final_permutation, [x for x in range(len(list_of_permutes[0].order))]):
                    first_data_node, last_data_node = list_of_permutes[0].in_node(), list_of_permutes[-1].out_node()
                else:
                    if len(list_of_permutes) < 2:
                        continue
                    first_data_node, last_data_node = list_of_permutes[0].out_node(), list_of_permutes[-1].out_node()
                    list_of_permutes[0].order = final_permutation

                graph.remove_edge(first_data_node.id, first_data_node.out_node().id)
                graph.remove_edge(last_data_node.in_node().id, last_data_node.id)

                merge_data_nodes(graph, first_data_node, last_data_node)
                graph.remove_node(last_data_node.id)
                graph_clean_up_tf(graph)
