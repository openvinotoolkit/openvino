# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.utils.error import Error


class ResizeOp(Op):
    enabled = False
    op = 'Resize'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,
            'op': self.op,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': self.resize_infer
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'mode',
        ]

    def backend_attrs(self):
        return [
            'mode',
        ]

    @staticmethod
    def resize_infer(node: Node):
        layout = node.graph.graph['layout']
        assert len(layout) == 4

        input_shape = node.in_node(0).shape
        if input_shape is None:
            raise Error('Input shape for operation "{}" is None'.format(node.soft_get('name', node.id)))

        scale_value = node.in_node(1).value

        node.out_port(0).data.set_shape(input_shape * scale_value)

