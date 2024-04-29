# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.activation_ops import Sin


class DummyExt1(FrontReplacementPattern):
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='ReLU'):
            new_node = Sin(graph, {'name': node.soft_get('name') + '/sin'}).create_node()
            node.out_port(0).get_connection().insert_node(new_node)
