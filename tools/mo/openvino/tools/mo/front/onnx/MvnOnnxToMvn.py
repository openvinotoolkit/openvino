# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.mvn import MVN
from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, rename_nodes


class MvnOnnxToMvn(FrontReplacementPattern):
    """
    Replace AttributedMVN operation from ONNX with MVN
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='MVNOnnx'):
            node_name = node.soft_get('name', node.id)

            new_mvn = create_op_with_const_inputs(graph, MVN, {1: node.axes},
                                                  {'eps': node.eps,
                                                   'eps_mode': node.eps_mode,
                                                   'normalize_variance': node.normalize_variance})
            node.in_port(0).get_connection().set_destination(new_mvn.in_port(0))
            node.out_port(0).get_connection().set_source(new_mvn.out_port(0))
            rename_nodes([(node, node_name + '/to_be_removed'), (new_mvn, node_name)])

            graph.remove_node(node.id)
