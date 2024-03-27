# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.Cast import Cast
from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.passes.convert_data_type import data_type_str_to_np


class ChangeRandomUniformOutputType(BackReplacementPattern):
    """
    This transformation adds Cast to IR data_type after RandomUniform operation
    when RandomUniform output type is not equal to IR data_type and RandomUniform output type
    is floating point type.
    'output_type' attribute determines the generation algorithm of RandomUniform, so output numbers
    generated for different values of 'output_type' may not be equal. For this reason 'output_type'
    attribute shouldn't be changed for matching of inference results. So in cases when we need
    to change the data type of RandomUniform we need to insert Cast node after RandomUniform.
    """
    enabled = True
    force_shape_inference = True

    def run_after(self):
        from openvino.tools.mo.back.MarkNodesWithShapeValues import MarkNodesWithShapeValues
        return [MarkNodesWithShapeValues]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        ir_data_type = data_type_str_to_np(graph.graph['cmd_params'].data_type)

        for node in graph.get_op_nodes(op='RandomUniform'):
            assert node.has_valid('output_type')

            if node.has_and_set('returns_shape_value'):
                continue

            if node.output_type != ir_data_type and np.issubdtype(node.output_type, np.floating):
                node_name = node.soft_get('name', node.id)
                convert_node = Cast(graph, {'name': node_name + "/cast", 'dst_type': ir_data_type}).create_node()
                node.out_port(0).get_connection().insert_node(convert_node)
