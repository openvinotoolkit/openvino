# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import undefined_shape_of_rank
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.ops.convolution import Convolution
from openvino.tools.mo.ops.op import Op


class DeformableConvolution(Op):
    op = 'DeformableConvolution'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': __class__.op,
            'op': __class__.op,
            'version': 'opset8',
            'infer': Convolution.infer,
            'group': 1,
            'deformable_group': 1,
            'multiplication_transparent': True,
            'multiplication_transparent_ports': [(0, 0), (2, 0)],
            'in_ports_count': 3,
            'out_ports_count': 1,
            'bilinear_interpolation_pad': False,
        }, attrs)

    def backend_attrs(self):
        # the same attributes as in a regular convolution and additional attributes 'deformable_group', 'group'
        # and 'bilinear_interpolation_pad'
        attrs = Convolution(self.graph, {}).backend_attrs() + ['deformable_group', 'group']
        if self.get_opset() == 'opset8':
            attrs.append('bilinear_interpolation_pad')
        return attrs

    @staticmethod
    def reverse_infer(node: Node):
        input_shape_1 = node.in_port(0).data.get_shape()
        input_shape_2 = node.in_port(1).data.get_shape()
        if input_shape_1 is None:
            node.in_port(0).data.set_shape(undefined_shape_of_rank(4))
        if input_shape_2 is None:
            node.in_port(1).data.set_shape(undefined_shape_of_rank(4))
