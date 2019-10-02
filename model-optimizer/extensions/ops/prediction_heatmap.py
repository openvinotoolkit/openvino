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
import numpy as np

from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class PredictionHeatmapOp(Op):
    op = 'PredictionHeatmap'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'infer': PredictionHeatmapOp.infer
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def infer(node: Node):
        in_node_shape = node.in_nodes()[0].shape.copy()
        top_shape = np.array([1, 1, 1, 1])
        num_person = in_node_shape[0]
        num_joints = in_node_shape[1]
        top_shape[2] = num_person
        top_shape[3] = 3 * num_joints
        node.out_node().shape = top_shape
