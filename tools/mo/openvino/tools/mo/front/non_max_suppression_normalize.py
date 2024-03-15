# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.reshape import Reshape


class NonMaxSuppressionNormalize(FrontReplacementSubgraph):
    """
    The transformation converts several inputs of the NonMaxSuppression layer to be 1D instead of 0D with shape [1] to
    comply with the layer specification.
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for nms in graph.get_op_nodes(op='NonMaxSuppression'):
            # make inputs 2 to 5 to have shape [1] instead of [0] (convert 0D to 1D)
            nms_name = nms.soft_get('name', nms.id)
            for port_id in range(2, 6):
                if port_id in nms.in_ports() and not nms.in_port(port_id).disconnected():
                    reshape_1d = create_op_node_with_second_input(graph, Reshape, int64_array([1]),
                                                                  {'name': nms_name + '/Reshape_1D_{}'.format(port_id)})
                    nms.in_port(port_id).get_connection().insert_node(reshape_1d)
