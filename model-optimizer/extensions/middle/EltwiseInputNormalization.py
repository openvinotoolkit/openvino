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
from copy import deepcopy

from extensions.middle.EltwiseInputReshape import EltwiseInputReshape
from mo.front.common.layout import get_features_dim, shape_for_layout
from mo.graph.graph import Node
from mo.middle.passes.fusing.helpers import get_value_id
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.op import Op
from mo.ops.reshape import Reshape


class EltwiseInputNormalize(EltwiseInputReshape, MiddleReplacementPattern):
    # This pass should be called directly from pipeline before layout change and other permutations
    enabled = False

    def find_and_replace_pattern(self, graph: nx.MultiDiGraph):
        eltwise_nodes = [Node(graph, node) for node in graph.node if Node(graph, node).soft_get('type') == 'Eltwise']
        # Iterating over all Eltwise operations and check that every input has similar shape
        # in case of different shapes, we inserts new_shape attribute and then call EltwiseInputReshape extension
        # that insert reshapes (in case of not constant nodes) or directly reshapes values in data nodes for specified
        # shape
        for node in eltwise_nodes:
            output_shape = node.out_node().shape
            for in_node in node.in_nodes().values():
                if len(in_node.shape) != len(output_shape):
                    # Set edge attribute new_shape for further transformation pass
                    new_shape = in_node.shape
                    for x in range(len(output_shape) - len(in_node.shape)):
                        new_shape = np.insert(new_shape, 0, 1)

                    nx.set_edge_attributes(G=node.graph,
                                           values={(in_node.id, node.id, 0): new_shape},
                                           name='new_shape')

        super().find_and_replace_pattern(graph)
