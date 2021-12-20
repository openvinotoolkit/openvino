# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from openvino.tools.mo.front.MatMul_normalizer import FullyConnectedDecomposer
from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.graph.graph import Graph, rename_nodes
from openvino.tools.mo.ops.MatMul import MatMul


class BatchDotReplacer(FrontReplacementPattern):
    """
    Replaces MXNet batch_dot with MatMul. Should run after FullyConnectedDecomposer, because batch_dot does not need
    in MatMul normalization.
    """
    enabled = True

    def run_after(self):
        return [FullyConnectedDecomposer]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='batch_dot'):
            node_name = node.soft_get('name', node.id)
            matmul_node = MatMul(graph, dict(name='MatMul', transpose_a=node.transpose_a,
                                             transpose_b=node.transpose_b)).create_node()
            node.in_port(0).get_connection().set_destination(matmul_node.in_port(0))
            node.in_port(1).get_connection().set_destination(matmul_node.in_port(1))
            node.out_port(0).get_connection().set_source(matmul_node.out_port(0))
            rename_nodes([(matmul_node, node_name)])
