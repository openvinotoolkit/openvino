"""
 Copyright (C) 2020 Intel Corporation

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

import numpy as np

from extensions.ops.non_zero import NonZero
from extensions.ops.transpose import Transpose
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementOp
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Node, Graph, rename_nodes


class WhereDecomposition(FrontReplacementOp):
    """
    This transformation decomposes the TF layer Where (when x = None, y = None) using the formula
        Where(condition) = Transpose(NonZero(condition), [1, 0])
    """
    op = 'Where'
    enabled = True

    def run_after(self):
        from extensions.front.tf.sparse_weighted_sum import ExperimentalSparseWeightedSumFrontReplacer
        from extensions.front.TransposeOrderNormalizer import TransposeOrderNormalizer
        return [ExperimentalSparseWeightedSumFrontReplacer, TransposeOrderNormalizer]

    def replace_op(self, graph: Graph, node: Node):
        node_name = node.soft_get('name', node.id)
        non_zero_node = NonZero(graph, {'name': node_name + '/NonZero_', 'output_type': np.int64}).create_node()
        transpose_node = create_op_node_with_second_input(graph, Transpose, int64_array([1, 0]), op_attrs={})
        non_zero_node.out_port(0).connect(transpose_node.in_port(0))
        rename_nodes([(node, node_name + '/delete'), (transpose_node, node_name)])

        non_zero_node.in_port(0).connect(node.in_port(0).get_source())
        return [transpose_node.id]
