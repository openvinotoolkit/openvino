# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.activation_ops import Tanh


class DummyExt2(FrontReplacementPattern):
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='Sigmoid'):
            new_node = Tanh(graph, {'name': node.soft_get('name') + '/tanh'}).create_node()
            node.out_port(0).get_connection().insert_node(new_node)
