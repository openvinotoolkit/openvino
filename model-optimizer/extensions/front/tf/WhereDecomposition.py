# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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
        from extensions.front.tf.embedding_segments_sum import EmbeddingSegmentsSumFrontReplacer, EmbeddingSegmentsSumFrontReplacer2
        from extensions.front.TransposeOrderNormalizer import TransposeOrderNormalizer
        return [EmbeddingSegmentsSumFrontReplacer, EmbeddingSegmentsSumFrontReplacer2, TransposeOrderNormalizer]

    def replace_op(self, graph: Graph, node: Node):
        node_name = node.soft_get('name', node.id)
        non_zero_node = NonZero(graph, {'name': node_name + '/NonZero_', 'output_type': np.int64}).create_node()
        transpose_node = create_op_node_with_second_input(graph, Transpose, int64_array([1, 0]), op_attrs={})
        non_zero_node.out_port(0).connect(transpose_node.in_port(0))
        rename_nodes([(node, node_name + '/delete'), (transpose_node, node_name)])

        non_zero_node.in_port(0).connect(node.in_port(0).get_source())
        return [transpose_node.id]
