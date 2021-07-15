# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph
from mo.middle.passes.convert_data_type import data_type_str_to_np


class ChangeRandomUniformOutputType(BackReplacementPattern):
    """
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

            if node.output_type != np.float16 and ir_data_type == np.float16:
                log.warning('Change data type from {} to {} for node {}'.format(node.output_type, ir_data_type, node.name))
                node.output_type = ir_data_type