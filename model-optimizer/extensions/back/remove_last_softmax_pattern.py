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

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph
from mo.middle.passes.eliminate import remove_op_node_with_data_node


class RemoveLastSoftMaxPattern(BackReplacementPattern):
    # This replacer is intentionally disabled and must be called if the flag --remove_output_softmax was enabled
    enabled = False

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('softmax_node', dict(op='SoftMax')),
                ('softmax_data', dict(kind='data')),
                ('op_output', dict(op='Result'))
            ],
            edges=[
                ('softmax_node', 'softmax_data'),
                ('softmax_data', 'op_output')
            ]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        """
        Removes output SoftMax layer
        :param graph: graph to operate on
        :param match: dictionary with matched nodes
        """
        if len(match['softmax_data'].out_nodes()) == 1:
            remove_op_node_with_data_node(graph, match['softmax_node'])
