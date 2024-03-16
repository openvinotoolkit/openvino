# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.ReduceOps import ReduceSum
from openvino.tools.mo.ops.elementwise import Pow
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.reshape import Reshape


class ReplacePNormNodePattern(MiddleReplacementPattern):
    """
    PNorm operation should be replaced by operations: Power(P) -> Reshape(n,c*g->n,g,c)-> ReduceSum(axis=1)-> Power(1/P)
    """
    enabled = True

    @staticmethod
    def pattern():
        return dict(
            nodes=[('op', dict(op='pnorm'))],
            edges=[])

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['op']
        shape = node.in_port(0).data.get_shape().copy()

        assert shape[1] % node.group == 0

        power_node = create_op_node_with_second_input(graph, Pow, node.p, {'name': node.id + '_power'})

        reshape_node = create_op_node_with_second_input(graph, Reshape,
                                                        int64_array([shape[0], shape[1] / node.group, node.group]),
                                                        {'name': node.id + '_reshape'})
        reshape_node.in_port(0).connect(power_node.out_port(0))

        reducesum_node = create_op_node_with_second_input(graph, ReduceSum,
                                                          int64_array([2]),
                                                          {'name': node.id + '_sum', 'keep_dims': False})
        reducesum_node.in_port(0).connect(reshape_node.out_port(0))

        invpower_node = create_op_node_with_second_input(graph, Pow, 1.0 / node.p, {'name': node.id + '_invpower'})

        invpower_node.in_port(0).connect(reducesum_node.out_port(0))

        node.in_port(0).get_connection().set_destination(power_node.in_port(0))
        node.out_port(0).get_connection().set_source(invpower_node.out_port(0))
