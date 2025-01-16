# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.squeeze import Squeeze


class GatherTreeNormalizer(BackReplacementPattern):
    enabled = True
    force_clean_up = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(type='GatherTree'):
            name = node.soft_get('name', node.id)
            assert 3 in node.in_ports() and not node.in_port(3).disconnected()

            end_token_shape = node.in_port(3).data.get_shape()
            assert end_token_shape is not None
            if end_token_shape.size == 1 and end_token_shape.ndim == 1:
                squeeze = create_op_node_with_second_input(graph, Squeeze, int64_array([0]),
                                                           {'name': name + '/Squeeze', 'override_output_shape': True})
                node.in_port(3).get_connection().insert_node(squeeze)
