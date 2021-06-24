# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph
from mo.middle.passes.convert_data_type import data_type_str_to_np


class ChangeCastOutputType(BackReplacementPattern):
    """
    Change the Cast dst_type from fp64 to fp32 since not all plugins support fp64 data type.
    Change the Cast dst_type from fp32 to fp16 when generating IR for fp16.
    But leave fp32 if node returns shape value even if --data_type=FP16 (look extensions/back/MarkNodesWithShapeValues.py).
    """
    enabled = True
    force_shape_inference = True

    def run_after(self):
        from extensions.back.MarkNodesWithShapeValues import MarkNodesWithShapeValues
        return [MarkNodesWithShapeValues]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='Cast'):
            if node.dst_type == np.float64:
                log.warning('Change data type from {} to {} for node {}'.format(node.dst_type, np.float32, node.name))
                node.dst_type = np.float32

            ir_data_type = data_type_str_to_np(node.graph.graph['cmd_params'].data_type)
            if node.dst_type == np.float32 and ir_data_type == np.float16 and not node.has_and_set('returns_shape_value'):
                log.warning('Change data type from {} to {} for node {}'.format(node.dst_type, ir_data_type, node.name))
                node.dst_type = ir_data_type
            elif node.has_and_set('returns_shape_value') and node.dst_type == np.float16:
                # return back FP32 for all Convert nodes with shape values
                log.warning('Change data type from {} to {} for node {} in ShapeOf subgraph'.
                            format(node.dst_type, np.float32, node.name))
                node.dst_type = np.float32
