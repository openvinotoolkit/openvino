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

from extensions.front.mxnet.ssd_pattern_flatten_softmax_activation import SsdPatternFlattenSoftmaxActivation
from extensions.front.mxnet.ssd_pattern_remove_flatten import SsdPatternRemoveFlatten
from extensions.front.mxnet.ssd_pattern_remove_reshape import SsdPatternRemoveReshape
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph


class SsdPatternRemoveTranspose(FrontReplacementSubgraph):
    enabled = True

    def run_before(self):
        return [SsdPatternFlattenSoftmaxActivation, SsdPatternRemoveFlatten, SsdPatternRemoveReshape]

    def pattern(self):
        return dict(
            nodes=[
                ('transpose', dict(op='Transpose')),
                ('softmax_activation', dict(op='SoftMax')),
                ('multi_box_detection', dict(op='_contrib_MultiBoxDetection'))
            ],
            edges=[
                ('transpose', 'softmax_activation', {'in': 0}),
                ('softmax_activation', 'multi_box_detection', {'in': 1}),
            ]
        )

    def replace_sub_graph(self, graph: Graph, match: dict):
        """
        Need to find each occurrence of pattern:
        transpose -> SoftmaxActivation -> _contrib_MultiBoxDetection
        remove transpose layer to secure the order of weights in SoftMax to be the same as IE expects
        IE expects weights to be in following order: class-wise values for each priorbox.
        priorboxes change the quickest

        Parameters
        ----------
        graph : Graph
           Graph with loaded model.
         match : dict
           Patterns which were found in graph structure.
        """
        transpose_node = match['transpose']
        softmax_activation = match['softmax_activation']
        transpose_in_node = transpose_node.in_node(0)

        graph.remove_edge(transpose_in_node.id, transpose_node.id)
        graph.remove_edge(transpose_node.id, softmax_activation.id)
        graph.remove_node(transpose_node.id)
        graph.create_edge(transpose_in_node, softmax_activation)
