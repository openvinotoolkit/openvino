# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.ops.pooling import Pooling


class AdaptiveAvgPooling(Op):
    '''
    Non-reshape-able op.
    '''
    enabled = False
    op = 'AdaptiveAvgPooling'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': None,
            'op': self.op,
            'infer': __class__.infer,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    @classmethod
    def infer(cls, node: Node):
        input_shape = node.in_node(0).shape
        input_h = input_shape[2]
        input_w = input_shape[3]
        output_h = node.output_size[0]
        output_w = node.output_size[1]

        stride_h = input_h // output_h
        stride_w = input_w // output_w
        kernel_h = input_h - (output_h - 1) * stride_h
        kernel_w = input_w - (output_w - 1) * stride_w

        data = {
            'window': int64_array([1, 1, kernel_h, kernel_w]),
            'stride': int64_array([1, 1, stride_h, stride_w]),
            'pad': int64_array([[0, 0], [0, 0], [0, 0], [0, 0]]),
            'pad_spatial_shape': int64_array([[0, 0], [0, 0]]),
            'pool_method': 'avg',
            'exclude_pad': False,
            'output_spatial_shape': None,
            'spatial_dims': None,
            'channel_dims': int64_array([1]),
            'batch_dims': int64_array([0]),
            'layout': 'NCHW',
            'rounding_type': 'floor',
            'pooling_convention': 'valid'
        }

        # update the attributes of the node
        Pooling.update_node_stat(node, data)
        Pooling.infer(node)
