# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import shape_array
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import PermuteAttrs, Op


class SwapAxis(Op):
    op = 'SwapAxis'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'infer': self.infer,
            'reverse_infer': self.reverse_infer,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    @staticmethod
    def infer(node: Node):
        node['order'] = list(range(node.in_node().shape.size))
        node.order[node.dim2], node.order[node.dim1] = node.order[node.dim1], node.order[node.dim2]

        input_shape = node.in_port(0).data.get_shape().copy()
        node.out_port(0).data.set_shape(input_shape[node.order])
        if node.in_port(0).data.get_value() is not None:
            node.out_port(0).data.set_value(np.transpose(node.in_port(0).data.get_value(), axes=node.order))

        PermuteAttrs.create_permute_attrs(node, attrs=[('order', 'input:0')])

    @staticmethod
    def reverse_infer(node: Node):
        output_shape = node.out_port(0).data.get_shape()
        if node.in_port(0).data.get_shape() is None and output_shape is not None:
            input_shape = output_shape.data.copy()
            input_shape[node.dim2], input_shape[node.dim1] = input_shape[node.dim1], input_shape[node.dim2]
            node.in_port(0).data.set_shape(shape_array(input_shape))
