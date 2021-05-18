# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from extensions.middle.ConvertLayoutDependentOperations import ConvertLayoutDependentOperations
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.reshape import Reshape


class TFDepthwiseConv2dNativeReshape(MiddleReplacementPattern):

    def run_before(self):
        return [ConvertLayoutDependentOperations]

    enabled = True
    force_clean_up = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op="DepthwiseConv2dNative"):
            node_name = node.soft_get('name', node.id)
            reshape_node = create_op_node_with_second_input(graph, Reshape, int64_array([0, 0, -1, 1]),
                                                            op_attrs=dict(name=node_name + '/Reshape',
                                                                          override_output_shape=True))
            node.in_port(1).get_connection().insert_node(reshape_node)
