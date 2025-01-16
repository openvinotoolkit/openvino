# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.layout import shape_for_layout, get_height_dim, get_batch_dim, get_features_dim, get_width_dim
from openvino.tools.mo.front.common.partial_infer.utils import dynamic_dimension, is_fully_defined
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.utils.error import Error


class DepthToSpaceOp(Op):
    op = 'DepthToSpace'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': self.op,
            'type': self.op,
            'version': 'opset1',

            'mode': 'blocks_first',

            'infer': self.infer,

            'in_ports_count': 1,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return ['mode', 'block_size']

    @staticmethod
    def infer(node: Node):
        in_shape = node.in_port(0).data.get_shape()
        if in_shape.size != 4:
            raise Error('TensorFlow DepthToSpace operation is supported for 4D \'NHWC\' input layout only. '
                        'Current input shape is \'{}\''.format(in_shape))

        layout = node.graph.graph['layout']

        N = in_shape[get_batch_dim(layout, 4)]
        H = in_shape[get_height_dim(layout, 4)]
        W = in_shape[get_width_dim(layout, 4)]
        C = in_shape[get_features_dim(layout, 4)]

        block_size = node['block_size']
        if C is not dynamic_dimension and C % (block_size ** 2):
            raise Error('Feature dimensions of input tensor of DepthToSpace operation have to be divisible by square '
                        'of DepthToSpace \'block_size\' parameter. Input tensor shape = {}. Feature dimension = {}. '
                        'block_size = {}'.format(in_shape, C, block_size))

        out_shape = shape_for_layout(layout,
                                     batch=N,
                                     features=C // (block_size * block_size),
                                     height=H * block_size,
                                     width=W * block_size)

        if is_fully_defined(in_shape) and is_fully_defined(out_shape) and np.prod(in_shape) != np.prod(out_shape):
            raise Error('Number of input elements "{}" is not equal to number of output elements "" for node "{}"'
                        ''.format(in_shape, out_shape, node.soft_get('name', node.id)))
        node.out_port(0).data.set_shape(out_shape)
