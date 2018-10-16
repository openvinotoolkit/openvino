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

from mo.front.subgraph_matcher import SubgraphMatch
from mo.front.tf.replacement import FrontReplacementFromConfigFileSubGraph
from mo.graph.graph import insert_node_after
from mo.ops.permute import Permute


class ConvFlattenReplacement(FrontReplacementFromConfigFileSubGraph):
    replacement_id = 'ConvFlatten'

    def output_edges_match(self, graph: nx.DiGraph, match: SubgraphMatch, new_sub_graph: dict):
        return {}

    def input_edges_match(self, graph: nx.DiGraph, match: SubgraphMatch, new_sub_graph: dict):
        return {}

    def nodes_to_remove(self, graph: nx.MultiDiGraph, match: SubgraphMatch):
        # no need to remove any of matched nodes. We just insert 'Permute' node before the matched sub-graph.
        return []

    def generate_sub_graph(self, graph: nx.MultiDiGraph, match: SubgraphMatch):
        permute_op = Permute(graph, {'order': np.array([0, 2, 3, 1])})
        permute_node = permute_op.add_node({'name': match.scope + '_permute_'})

        reshape_node = match.node_by_pattern('flatten/Reshape$')

        # reshape_in_node is the node after which we should insert Permute
        reshape_in_node = reshape_node.in_nodes()[0]
        insert_node_after(reshape_in_node, permute_node, 0)
        return {}
