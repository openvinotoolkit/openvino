# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.middle.passes.convert_data_type import data_type_str_to_np
from openvino.tools.mo.utils.error import Error

operations_with_data_type_attributes = {
    'Cast': {'attr_name': 'dst_type', 'in_ports_to_check': (0,)},
    'Range': {'attr_name': 'output_type', 'in_ports_to_check': (0, 1, 2)},
}


class ChangeOutputTypeAttributes(BackReplacementPattern):
    """
    The transformation changes output type for the specific operations defined in the
    operations_with_data_type_attributes dictionary if one of the following conditions is met:
    - The operation output type is fp64. Since not all plugins support fp64 data type it is converted to fp32.
    - Changes output type from fp32 to fp16 (and ensure that this is possible) when generating fp16 IR.
    - Keep operation output type equal to fp32 for operations located in the shape calculation sub-graphs to
    avoid floating point overflow.
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

        for node in graph.get_op_nodes():
            if node.op in operations_with_data_type_attributes:
                dst_type = operations_with_data_type_attributes[node.op]['attr_name']
                node_name = node.soft_get('name', node.id)
                assert node.has_valid(dst_type), '{} attribute is missing for node {}'.format(dst_type, node_name)

                final_type = None
                if node[dst_type] == np.float64:
                    final_type = np.float32

                if node[dst_type] in [np.float32, np.float64] and ir_data_type == np.float16 and \
                        not node.has_and_set('returns_shape_value'):
                    final_type = np.float16
                elif node.has_and_set('returns_shape_value') and node[dst_type] == np.float16:
                    # return back FP32 for all nodes with shape values
                    final_type = np.float32

                if final_type is not None:
                    log.warning('Change data type from {} to {} for node {}'.format(node[dst_type], final_type,
                                                                                    node_name))
                    node[dst_type] = final_type

                if final_type == np.float16:
                    assert_that_is_castable_to_fp16(node)


def assert_that_is_castable_to_fp16(node: Node):
    op_name = node.soft_get('op')
    node_name = node.soft_get('name', node.id)

    for i in operations_with_data_type_attributes[op_name]['in_ports_to_check']:
        val = node.in_port(i).data.get_value()
        if val is None:
            return

        if np.any(val > np.finfo(np.float16).max) or np.any(val < np.finfo(np.float16).min):
            raise Error("Try to convert with --data_type=FP32 argument. "
                        "This model can not be converted to FP16 precision, since "
                        "'{}' node value {} exceeds FP16 allowed limits: [{}, {}]"
                        .format(node_name, val, np.finfo(np.float16).min, np.finfo(np.float16).max))
        # further this input values will be rewritten since force_shape_inference=True
        node.in_port(i).data.set_value(val.astype(np.float16))

    original_output = node.out_port(0).data.get_value()
    node.infer(node)
    casted_output = node.out_port(0).data.get_value()
    original_output_len = original_output.size if hasattr(original_output, 'size') else None
    casted_output_len = casted_output.size if hasattr(casted_output, 'size') else None

    if original_output_len != casted_output_len:
        raise Error("Try to convert with --data_type=FP32 argument. "
                    "This model can not be converted to FP16 precision, since "
                    "after conversion of '{}' node to FP16 output shape {} differs from the original {}."
                    .format(node_name, casted_output_len, original_output_len))

    diff_count = np.count_nonzero(np.subtract(original_output, casted_output) > 1.e-4)
    if diff_count > 0:
        log.warning("{} elements of {} of Range node '{}' output differ from the original values while "
                    "converting network to FP16 precision".format(diff_count, len(original_output), node_name))
