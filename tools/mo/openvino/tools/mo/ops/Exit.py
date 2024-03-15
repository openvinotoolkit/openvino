# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class Exit(Op):
    op = "Exit"

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,
            'op': __class__.op,
            'infer': Exit.exit_infer,
            'in_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def exit_infer(node: Node):
        output_shape = node.in_port(0).data.get_shape()
        output_value = node.in_port(0).data.get_value()

        for port in node.out_ports():
            if not node.out_port(port).disconnected():
                node.out_port(port).data.set_shape(output_shape)
                if output_value is not None:
                    node.out_port(port).data.set_value(output_value)
