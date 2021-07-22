# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.Cast import Cast
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph
from mo.middle.passes.convert_data_type import data_type_str_to_np


class ChangeRandomUniformOutputType(BackReplacementPattern):
    """
    This transformation adds Convert to IR data_type after RandomUniform operation
    when RandomUniform output type is not equal to IR data_type and RandomUniform output type
    is floating point type.
    """
    enabled = True
    force_shape_inference = True

    def run_after(self):
        from extensions.back.MarkNodesWithShapeValues import MarkNodesWithShapeValues
        return [MarkNodesWithShapeValues]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        ir_data_type = data_type_str_to_np(graph.graph['cmd_params'].data_type)

        for node in graph.get_op_nodes(op='RandomUniform'):
            assert node.has_valid('output_type')

            if node.output_type != ir_data_type and np.issubdtype(node.output_type, np.floating):
                node_name = node.soft_get('name', node.id)
                convert_node = Cast(graph, {'name': node_name + "/convert", 'dst_type': ir_data_type}).create_node()
                node.out_port(0).get_connection().insert_node(convert_node)
