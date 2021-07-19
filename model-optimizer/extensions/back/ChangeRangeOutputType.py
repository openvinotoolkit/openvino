# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph, Node
from mo.middle.passes.convert_data_type import data_type_str_to_np
from mo.utils.error import Error


class ChangeRangeOutputType(BackReplacementPattern):
    """
    Change Range 'output_type' from fp64 to fp32 since not all plugins support fp64 data type.
    And from fp32 to fp16 when generating IR for fp16. Before changing precision to FP16 ensures that it's possible
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

        for node in graph.get_op_nodes(op='Range'):
            node_name = node.soft_get('name', node.id)
            assert node.has_valid('output_type')

            final_type = None
            if node.output_type == np.float64:
                final_type = np.float32
            if node.output_type in [np.float32, np.float64] and ir_data_type == np.float16:
                final_type = np.float16

            if final_type == np.float16:
                assert_that_is_castable_to_fp16(node)

            if final_type is not None:
                node.output_type = final_type
                log.warning('Change output_type from {} to {} for Range node {}'.format(
                    node.output_type, final_type, node_name))


def assert_that_is_castable_to_fp16(node: Node):
    node_name = node.soft_get('name', node.id)
    start, limit, delta = [node.in_port(i).data.get_value() for i in range(3)]
    for val in [start, limit, delta]:
        if val > np.finfo(np.float16).max or val < np.finfo(np.float16).min:
            raise Error("This model can not be converted to FP16 precision, since "
                        "Range node '{}' input value {} exceeds FP16 allowed limits: [{}, {}]"
                        .format(node_name, val, np.finfo(np.float16).min, np.finfo(np.float16).max))

    start, limit, delta = [node.in_port(i).data.get_value().astype(np.float16) for i in range(3)]
    casted_output = np.arange(start, limit, delta, dtype=node['output_type'])
    original_output = node.out_port(0).data.get_value()
    if len(original_output) != len(casted_output):
        raise Error("This model can not be converted to FP16 precision, since "
                    "after changing Range node '{}' dtype to FP16 output shape {} differs from original {}"
                    .format(node_name, len(casted_output), len(original_output)))

    diff_count = np.count_nonzero(np.subtract(original_output, casted_output) > 1.e-4)
    if diff_count > 0:
        log.warning("{} elements of {} of Range node '{}' output differ from the original values while "
                    "converting network to FP16 precision".format(diff_count, len(original_output), node_name))
