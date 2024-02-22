# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.graph.graph import Graph, rename_node
from openvino.tools.mo.ops.eye import Eye
from openvino.tools.mo.utils.error import Error


class EyeTFToEye(FrontReplacementPattern):
    """
    This transformation converts TFEye operation (TensorFlow semantic) to Eye operation (OpenVINO semantic).
    Refer to the Op implementation for the operations semantics description.
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for tfeye in graph.get_op_nodes(op='TFEye'):
            # save the original node name to use it in the new Eye op instance
            original_name = tfeye.soft_get('name', tfeye.id)
            tfeye['name'] = original_name + '/to_be_removed'

            if not tfeye.has_valid('output_type'):
                raise Error("TFEye should have valid ''output_type'' attribute.")
            output_type = tfeye.soft_get('output_type')

            new_eye = Eye(graph, {'output_type': output_type}).create_node()
            rename_node(new_eye, original_name)

            # num_rows
            tfeye.in_port(0).get_connection().set_destination(new_eye.in_port(0))
            # num_columns
            if not tfeye.in_port(1).disconnected:
                tfeye.in_port(1).get_connection().set_destination(new_eye.in_port(1))
            # batch_shape
            if not tfeye.in_port(2).disconnected:
                tfeye.in_port(2).get_connection().set_destination(new_eye.in_port(3))

            diagonal_index = Const(graph, {'name': original_name + '/diagonal_index',
                                    'value': 0}).create_node()
            diagonal_index.out_port(0).connect(new_eye.in_port(2))

            tfeye.out_port(0).get_connection().set_source(new_eye.out_port(0))
            graph.remove_node(tfeye.id)
