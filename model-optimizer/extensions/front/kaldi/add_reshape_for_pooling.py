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
import copy

import networkx as nx
import numpy as np

from extensions.front.kaldi.replace_lstm_node_pattern import create_node
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Node


class ReplacePoolingReshape(FrontReplacementOp):
    op = "Pooling"
    enabled = True

    def replace_op(self, graph: nx.MultiDiGraph, node: Node):
        input_nodes = node.in_nodes()

        pool_attrs = graph.node[node.id]['pb'].__dict__
        input_reshape = create_node(graph, 'Reshape_Pooling', {'type': 'Reshape',
                                                               'axis': 1,
                                                               'num_axes': -1,
                                                               'dim': None},
                                    tuple(n.id for i, n in input_nodes.items()))

        pooling = create_node(graph, 'pooling', pool_attrs, tuple([input_reshape.id]))

        output_reshape = create_node(graph, 'Pooling_Reshape', {
            'type': 'Reshape',
            'axis': 1,
            'num_axes': -1,
            'dim': None
        }, tuple([pooling.id]))

        return [output_reshape.id]
