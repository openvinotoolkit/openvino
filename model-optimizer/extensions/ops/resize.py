# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class ResizeOp(Op):
    enabled = False
    op = 'Resize'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,
            'op': __class__.op,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': __class__.resize_infer
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
            return

        scale_value = node.in_node(1).value

        node.out_node().shape = int64_array(input_shape * scale_value)

