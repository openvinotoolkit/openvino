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

from extensions.middle.ShufflenetReshape import FeatureShuffleReshape
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.permute import Permute
from mo.ops.reshape import Reshape
from mo.utils.error import Error


class ShuffleChannel(MiddleReplacementPattern):
    """
    Replaces Caffe ShuffleChannel with Reshapes and Permute layers
    """

    enabled = True

    def run_after(self):
        return [FeatureShuffleReshape]

    def pattern(self):
        return dict(
            nodes=[
                ('op', dict(op='ShuffleChannel')),
            ],
            edges=[
            ])

    def replace_pattern(self, graph: nx.MultiDiGraph, match: dict):
        if graph.graph['layout'] != "NCHW":
            return

        node = match['op']

        in_node = node.in_node(0)
        out_node = node.out_node(0)
        group = int(node['group'])

        graph.remove_edge(in_node.id, node.id)
        graph.remove_edge(node.id, out_node.id)

        rows = group
        cols = in_node.shape[1] // group

        if rows * cols != in_node.shape[1]:
            raise Error("Group {} should divide input channels number {} without reminder for node {}".format(group, in_node.shape[1], node.id))

        reshape_split = Reshape(graph, attrs={'name': node.id + '/Reshape_split_',
                                              'dim': np.array([in_node.shape[0], rows, cols, -1])})
        reshape_split_node = reshape_split.create_node_with_data([in_node])
        transpose = Permute(graph, attrs={'name': node.id + '/Transpose_',
                                          'order': np.array([0, 2, 1, 3])})
        transpose_node = transpose.create_node_with_data([reshape_split_node])
        reshape_concat = Reshape(graph, attrs={'name': node.id + '/Reshape_concat_',
                                               'dim': out_node.shape})
        reshape_concat.create_node_with_data([transpose_node], data_nodes=[out_node])
