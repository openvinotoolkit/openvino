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

from mo.front.common.replacement import FrontReplacementOp
from mo.ops.const import Const
from mo.ops.lin_op import Mul, Add


class ImageScaler(FrontReplacementOp):
    op = "ImageScaler"
    enabled = True

    def replace_sub_graph(self, graph: nx.MultiDiGraph, match: dict):
        # This replacer replace ImageScalar operation to Mul->Add sequence
        # Also it check that weights and biases are good
        op = match['op']

        # Check that weights and biases are not useless
        has_bias, has_weights = True, True
        if all([x == 1 for x in np.nditer(op.scale)]):
            has_weights = False
        if all([x == 0 for x in np.nditer(op.bias)]):
            has_bias = False

        # Get all outputs for op node
        out_nodes = [node for node in op.out_nodes().values()]

        assert len(op.in_nodes()) == 1

        last_node = op.in_node()
        # Create Mul & Add nodes
        if has_weights:
            mul_weights = Const(graph, dict(value=op.scale, shape=op.scale.shape))
            mul_op = Mul(graph, dict(name=op.id + '/mul_'))
            last_node = mul_op.create_node(inputs=[last_node, mul_weights.create_node()])

        if has_bias:
            add_bias = Const(graph, dict(value=op.bias, shape=op.bias.shape))
            add_op = Add(graph, dict(name=op.id + '/add_'))
            last_node = add_op.create_node(inputs=[last_node, add_bias.create_node()])

        # Move edges from ImageScaler to last_node (Mul or Add)
        for out_node in out_nodes:
            edge_attrs = graph.get_edge_data(op.id, out_node.id)[0]
            graph.remove_edge(op.id, out_node.id)
            graph.add_edges_from([(last_node.id, out_node.id, edge_attrs)])

        # Disconnect ImageScalar node
        graph.remove_edge(op.in_node().id, op.id)
