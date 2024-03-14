# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math

from openvino.tools.mo.front.common.layout import get_batch_dim, get_features_dim, get_height_dim, get_width_dim, shape_for_layout
from openvino.tools.mo.front.common.partial_infer.utils import dynamic_dimension, shape_array, dynamic_dimension_value
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class UpsampleOp(Op):
    op = 'Upsample'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': self.op,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': UpsampleOp.upsample_infer
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'height_scale',
            'width_scale',
            'mode',
        ]

    @staticmethod
    def upsample_infer(node: Node):
        node_name = node.soft_get('name', node.id)
        layout = node.graph.graph['layout']
        assert len(layout) == 4, 'Input tensor rank must be equal to 4 for node "{}"'.format(node_name)

        input_shape = node.in_port(0).data.get_shape()

        if len(node.in_nodes()) == 1:
            in_height = input_shape[get_height_dim(layout, 4)]
            in_width = input_shape[get_width_dim(layout, 4)]
            assert node.has('width_scale') is not None and node.has('height_scale') is not None
            if in_height is not dynamic_dimension:
                out_height = math.floor(in_height * node.height_scale)
            else:
                out_height = dynamic_dimension
            if in_width is not dynamic_dimension:
                out_width = math.floor(in_width * node.width_scale)
            else:
                out_width = dynamic_dimension
            node.out_port(0).data.set_shape(shape_for_layout(layout,
                                                             batch=input_shape[get_batch_dim(layout, 4)],
                                                             features=input_shape[get_features_dim(layout, 4)],
                                                             height=out_height,
                                                             width=out_width))
        else:
            scales = node.in_port(1).data.get_value()
            assert scales is not None, 'The input with scales for node "{}" is not constant'.format(node_name)
            eps = 1e-5  # This is to make rounding in case of very close number to round to closest instead of down
            # generic output shape calculation to support 5D input shape case
            output_shape = shape_array([dynamic_dimension for _ in range(len(input_shape))])
            for idx in range(len(output_shape)):
                if input_shape[idx] is not dynamic_dimension:
                    output_shape[idx] = int((input_shape[idx] + eps) * scales[idx])
                else:
                    output_shape[idx] = dynamic_dimension_value
            node.out_port(0).data.set_shape(output_shape)
