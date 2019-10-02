"""
 Copyright (c) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import math

import numpy as np

from mo.front.common.partial_infer.elemental import single_output_infer
from mo.front.common.partial_infer.reshape import tf_reshape_shape_infer
from mo.graph.graph import Node, Graph
from mo.ops.op import Op
from mo.utils.error import Error


class Reshape(Op):
    op = 'Reshape'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'kind': 'op',
            'type': __class__.op,
            'op': __class__.op,
            'reinterp_shape': True,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': __class__.infer,
        }, attrs)

    @staticmethod
    def infer(node: Node):
        single_output_infer(
            node,
            tf_reshape_shape_infer,
            lambda node: np.reshape(node.in_node(0).value, node.out_node().shape)
        )

    @staticmethod
    def kaldi_infer(node: Node):
        in_node = node.in_node().in_node()  # prev_layer_node -> data -> this_node
        input_shape = node.in_node().shape
        # Kaldi Reshape hugely depends on the layers that precedes or succeeds
        # Convolution/Pooling layers. Therefore there are 4 cases with different
        # partial inference.
        batch = input_shape[0]
        if in_node.op in ['Convolution', 'Pooling', 'Transpose']:
            output_spatial = np.array([batch, np.prod(input_shape[1:])], dtype=np.int64)
            return Reshape.set_shape_and_dim(node, output_spatial)
        # Supports ONLY NCHW and NH layouts
        if len(input_shape) not in [4, 2]:
            raise Error('Reshape in Kaldi support only 1d or 3d shapes')
        spatial_shape = input_shape[1]
        if len(input_shape) in [4]:
            spatial_shape = input_shape[2:3]
        out_node = node.out_node().out_node()
        if out_node.op == 'Convolution':
            output_spatial = np.array(
                [batch, math.ceil(spatial_shape / out_node.patch_stride), 1, out_node.patch_stride], dtype=np.int64)
            return Reshape.set_shape_and_dim(node, output_spatial)
        elif out_node.op == 'Pooling':
            if out_node.pool_step is None:
                out_node.stride = np.array([1, 1, out_node.window[-1], out_node.window[-1]], dtype=np.int64)
            output_spatial = np.array(
                [batch, out_node.pool_stride, 1, math.ceil(spatial_shape / out_node.pool_stride)], dtype=np.int64)
            return Reshape.set_shape_and_dim(node, output_spatial)

    @staticmethod
    def set_shape_and_dim(node: Node, reshape_dim):
        Reshape.update_node_stat(node, {'dim': reshape_dim})
        node.out_node().shape = reshape_dim
