# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.passes.eliminate import remove_op_node_with_data_node
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern


class RemoveUselessPad(MiddleReplacementPattern):
    """
    The Pad layer is removed if all padding values are equal to 0 (Constant values).
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='Pad'):
            all_pads_zeros = True
            for in_port_ind in range(1, 3):
                input_node = node.in_port(in_port_ind).get_source().node
                value = input_node.soft_get('value', None)
                all_pads_zeros &= input_node.soft_get('type') == 'Const' and value is not None and np.all(value == 0)

            if all_pads_zeros:
                remove_op_node_with_data_node(graph, node)
