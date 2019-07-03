"""
 Copyright (c) 2017-2019 Intel Corporation

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

from extensions.front.mxnet.ssd_pattern_remove_reshape import SsdPatternRemoveReshape
from mo.graph.graph import Graph
from mo.front.common.replacement import FrontReplacementSubgraph


class SsdPatternRemoveFlatten(FrontReplacementSubgraph):
    enabled = True

    def run_before(self):
        return [SsdPatternRemoveReshape]

    def pattern(self):
        return dict(
            nodes=[
                ('multi_box_prior', dict(op='_contrib_MultiBoxPrior')),
                ('flatten', dict(op='Flatten'))
            ],
            edges=[
                ('multi_box_prior', 'flatten', {'in': 0})
            ]
        )

    def replace_sub_graph(self, graph: Graph, match: dict):
        """
        Need to find each occurrence of pattern: _contrib_MultiBoxPrior -> Flatten
        remove Flatten layer - IE does not expect outputs to be flattened

        Parameters
        ----------
        graph : Graph
           Graph with loaded model.
         match : dict
           Patterns which were found in graph structure.
        """
        graph.erase_node(match['flatten'])
