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
import copy

import networkx as nx
import numpy as np
from mo.graph.graph import Node
from mo.ops.op import Op, PermuteAttrs


class Split(Op):
    op = 'Split'
    enabled = True

    def __init__(self, graph: nx.MultiDiGraph, attrs: dict):
        super().__init__(graph, {
            'type': 'Split',
            'op': 'Split',
            'axis': 1,
            'input_port': 0,
            'infer': Split.infer
        }, attrs)

    def supported_attrs(self):
        return ['axis', 'num_split']

    @staticmethod
    def infer(node: Node):
        input_node = node.in_node(0)
        outputs = node.out_nodes()
        out_shape = copy.copy(input_node.shape)
        out_shape[node.axis] = np.int64(input_node.shape[node.axis] / node.pb.num_split)
        for idx, output in outputs.items():
            output.shape = out_shape
        PermuteAttrs.create_permute_attrs(node, attrs=[('axis', 'input:0')])

