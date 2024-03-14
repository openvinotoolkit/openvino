# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.reshape import Reshape


class ArgMaxFlatten(FrontReplacementOp):
    """
    The ArgMax layer in Caffe may have non-specified 'axis' attribute. In this case it should flatten input data before
    calculating ArgMax.
    """
    op = "ArgMax"
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        argmax_node = match['op']
        if not argmax_node.has_valid('axis'):
            flatten_node = create_op_node_with_second_input(graph, Reshape, int64_array([0, 1, -1]),
                                                            dict(name=argmax_node.name + '/Flatten'))
            argmax_node.in_port(0).get_connection().insert_node(flatten_node)
            argmax_node.axis = 2
