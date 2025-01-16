# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.layout import shape_for_layout, get_height_dim, get_batch_dim, get_features_dim, get_width_dim
from openvino.tools.mo.front.common.partial_infer.utils import dynamic_dimension, is_fully_defined
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.utils.error import Error


class SpaceToDepth(Op):
    op = 'SpaceToDepth'

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
        in_shape = node.in_node().shape
        if in_shape.size != 4:
            raise Error('TensorFlow SpaceToDepth operation is supported for 4D \'NHWC\' input layout only. '
                        'Current input shape is \'{}\''.format(in_shape))

        layout = node.graph.graph['layout']
        N = in_shape[get_batch_dim(layout, 4)]
        H = in_shape[get_height_dim(layout, 4)]
        W = in_shape[get_width_dim(layout, 4)]
        C = in_shape[get_features_dim(layout, 4)]

        block_size = node['block_size']
        if (H is not dynamic_dimension and H % block_size) or (W is not dynamic_dimension and W % block_size):
            raise Error('Spatial dimensions of input tensor of SpaceToDepth operation have to be divisible by '
                        'SpaceToDepth \'block_size\' parameter. Input tensor shape = {}. Spatial dimensions = {},{}. '
                        'block_size = {}'.format(in_shape, H, W, block_size))

        out_shape = shape_for_layout(layout,
                                     batch=N,
                                     features=C * (block_size ** 2),
                                     height=H // block_size,
                                     width=W // block_size)

        node.out_port(0).data.set_shape(out_shape)
