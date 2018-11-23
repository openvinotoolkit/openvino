"""
 Copyright (c) 2017-2018 Intel Corporation

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
from mo.front.common.replacement import FrontReplacementPattern


class CheckSoftmaxNodeInputs(FrontReplacementPattern):

    enabled = True

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('softmax', dict(op='SoftmaxOutput'))
            ],
            edges=[])

    @staticmethod
    def replace_pattern(graph: nx.MultiDiGraph, match: dict):
        """
        Need to remove from softmax layer all unused inputs
        Parameters
        ----------
        graph : nx.MultiDiGraph
           Graph with loaded model.
         match : dict
           Patterns which were found in graph structure.
        """

        softmax_node = match['softmax']
        softmax_nodes_len = len(softmax_node.in_nodes())
        for i in reversed(range(1, softmax_nodes_len)):
            in_node = softmax_node.in_node(i)
            graph.remove_edge(in_node.id, softmax_node.id)
            graph.remove_node(in_node.id)
