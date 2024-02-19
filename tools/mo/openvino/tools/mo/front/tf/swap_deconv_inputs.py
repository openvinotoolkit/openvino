# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.graph.graph import Graph


class SwapDeconvInputs(FrontReplacementSubgraph):
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(swap_0_and_2_inputs=True):
            shape_src = node.in_port(0).get_source()
            node.in_port(0).disconnect()

            node.in_port(2).get_connection().set_destination(node.in_port(0))
            shape_src.connect(node.in_port(2))
            node['swap_0_and_2_inputs'] = False
