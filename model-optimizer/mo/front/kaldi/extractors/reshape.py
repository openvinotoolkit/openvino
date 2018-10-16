"""
 Copyright (c) 2018 Intel Corporation

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
import numpy as np

from mo.front.common.partial_infer.reshape import tf_reshape_shape_infer
from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import FrontExtractorOp
from mo.graph.graph import Node
from mo.ops.op import Op
from mo.ops.reshape import Reshape


class ReshapeFrontExtractor(FrontExtractorOp):
    op = 'reshape'
    enabled = True

    @staticmethod
    def extract(node):
        mapping_rule = {
            'dim': node.pb.dim if hasattr(node.pb, 'dim') else None,
            'axis': node.pb.axis,
            'num_axes': node.pb.num_axes,
            'infer': ReshapeFrontExtractor.infer
        }
        Op.get_op_class_by_name('Reshape').update_node_stat(node, mapping_rule)
        return __class__.enabled

    @staticmethod
    def infer(node: Node):
        in_node = node.in_node().in_node()  # prev_layer_node -> data -> this_node
        input_shape = node.in_node().shape
        # Kaldi Reshape hugely depends on the layers that precedes or succeeds
        # Convolution/Pooling layers. Therefore there are 4 cases with different
        # partial inference.
        batch = input_shape[0]
        if in_node.type == 'Convolution' or in_node.type == 'Pooling':
            output_spatial = int64_array([batch, np.prod(input_shape[1:])])
            return ReshapeFrontExtractor.set_shape_and_dim(node, output_spatial)
        # Supports ONLY NCHW and NH layouts
        spatial_shape = input_shape[1]
        if input_shape.shape == (4,):
            spatial_shape = input_shape[2:3]
        out_node = node.out_node().out_node()
        if out_node.type == 'Convolution':
            output_spatial = int64_array([batch, int(np.ceil(spatial_shape / out_node.patch_stride)), 1, out_node.patch_stride])
            return ReshapeFrontExtractor.set_shape_and_dim(node, output_spatial)
        elif out_node.type == 'Pooling':
            output_spatial = int64_array([batch, out_node.pool_stride, 1, int(np.ceil(spatial_shape / out_node.pool_stride))])
            return ReshapeFrontExtractor.set_shape_and_dim(node, output_spatial)

    @staticmethod
    def set_shape_and_dim(node: Node, reshape_dim):
        Reshape.update_node_stat(node, {'dim': reshape_dim})
        node.out_node().shape = reshape_dim

