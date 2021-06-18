# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.front.common.partial_infer.utils import int64_array, tf_window_op_pad_infer
from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class ExtractImagePatches(Op):
    op = "ExtractImagePatches"

    def __init__(self, graph: Graph, attrs: dict):
        assert 'spatial_dims' in attrs, \
            'ExtractImagePatches operation should have `spatial_dims` parameter set during creation'

        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': 'opset3',
            'infer': self.infer,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    def backend_attrs(self):
        return [
            ('sizes', lambda node: ','.join(map(str, node['sizes'][node.spatial_dims]))),
            ('strides', lambda node: ','.join(map(str, node['strides'][node.spatial_dims]))),
            ('rates', lambda node: ','.join(map(str, node['rates'][node.spatial_dims]))),
            'auto_pad',
        ]

    @staticmethod
    def infer(node: Node):
        assert [port.idx for port in node.in_ports().values() if not port.disconnected()] == [0], \
            'Wrong input nodes number for node {} with type ExtractImagePatches'.format(node.soft_get('name', node.id))
        input_shape = node.in_port(0).data.get_shape()
        name = node.soft_get('name', node.id)
        assert input_shape is not None, 'Input shape is not set for node {} with type ExtractImagePatches'.format(name)

        assert len(input_shape) == 4, 'ExtractImagePatches operation supports only 4D tensors'

        # the operation ExtractImagePatches should always calculate output shape in NHWC layout (because
        # in OV the operation produces the tensor in NHWC layout)
        N = input_shape[0]
        C = input_shape[3]

        size_spatial = int64_array(node.sizes)[node.spatial_dims]

        input_spatial_shape = input_shape[node.spatial_dims]
        stride_spatial_shape = node.strides[node.spatial_dims]

        size_extent = node.rates[node.spatial_dims] * (size_spatial - 1) + 1

        pad_spatial_shape, output_spatial_shape = tf_window_op_pad_infer(input_spatial_shape,
                                                                         size_extent,
                                                                         stride_spatial_shape,
                                                                         node.auto_pad,
                                                                         False)

        out_shape = [N, output_spatial_shape[0], output_spatial_shape[1], C * np.prod(size_spatial)]
        node.out_port(0).data.set_shape(int64_array(out_shape))
