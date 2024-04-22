# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.eye import Eye
from openvino.tools.mo.utils.error import Error


class EyeMXToEye(FrontReplacementPattern):
    """
    This transformation converts MXEye operation (MXNet semantic) to Eye operation (OpenVINO semantic).
    Refer to the Op implementation for the operations semantics description.
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for mxeye in graph.get_op_nodes(op='MXEye'):
            # save the original node name to use it in the new Eye op instance
            original_name = mxeye.soft_get('name', mxeye.id)
            mxeye['name'] = original_name + '/to_be_removed'

            if not mxeye.has_valid('num_rows'):
                raise Error("MXEye should have valid ''num_rows'' attribute.")
            num_rows = mxeye.soft_get('num_rows')

            if not mxeye.has_valid('num_columns'):
                raise Error("MXEye should have valid ''num_columns'' attribute.")
            num_columns = mxeye.soft_get('num_columns')

            if not mxeye.has_valid('diagonal_index'):
                raise Error("MXEye should have valid ''diagonal_index'' attribute.")
            diagonal_index = mxeye.soft_get('diagonal_index')
            
            if not mxeye.has_valid('output_type'):
                raise Error("MXEye should have valid ''output_type'' attribute.")
            output_type = mxeye.soft_get('output_type')

            new_eye = create_op_with_const_inputs(graph, Eye, {0: int64_array(num_rows),
                                                               1: int64_array(num_columns),
                                                               2: int64_array(diagonal_index)},
                                                              {'name': original_name + '/Gathered',
                                                               'output_type': output_type})
            mxeye.out_port(0).get_connection().set_source(new_eye.out_port(0))
            graph.remove_node(mxeye.id)
