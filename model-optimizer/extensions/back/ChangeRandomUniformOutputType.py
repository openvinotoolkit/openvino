# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph
from mo.middle.passes.convert_data_type import data_type_str_to_np


class ChangeRandomUniformOutputType(BackReplacementPattern):
    """
    Changes the RandomUniform output_type from fp32 or fp64 to fp16 when generating IR for fp16 and
    from fp16 or fp64 to fp32 when generating IR for fp32.
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
                log.warning('Change data type from {} to {} for node {}'.format(node.output_type, ir_data_type, node.name))
                node.output_type = ir_data_type
