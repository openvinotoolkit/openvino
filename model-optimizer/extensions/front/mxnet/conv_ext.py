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

from mo.front.common.extractors.utils import layout_attrs
from mo.front.extractor import FrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.ops.convolution import Convolution


class ConvFrontExtractor(FrontExtractorOp):
    op = 'Convolution'
    enabled = True

    @staticmethod
    def extract(node):
        attr = get_mxnet_layer_attrs(node.symbol_dict)

        kernel = attr.tuple("kernel", int, None)
        stride = attr.tuple("stride", int, (1, 1))
        padding = attr.tuple("pad", int, (0, 0))
        dilate = attr.tuple("dilate", int, (1, 1))
        group = attr.int("num_group", 1)
        output = attr.int("num_filter", None)
        bias_term = attr.str("no_bias", 'False') == 'False'

        node_attrs = {
            'op': 'Conv2D',
            'bias_addable': True,
            'bias_term': bias_term,
            'pad': np.array([[0, 0], [0, 0], [padding[0], padding[0]], [padding[1], padding[1]]], dtype=np.int64),
            'pad_spatial_shape': np.array([[padding[0], padding[0]], [padding[1], padding[1]]], dtype=np.int64),
            'dilation': np.array([1, 1, dilate[0], dilate[1]], dtype=np.int64),
            'output_spatial_shape': None,
            'output_shape': None,
            'stride': np.array([1, 1, stride[0], stride[1]], dtype=np.int64),
            'group': group,
            'output': output,
            'kernel_spatial': np.array([kernel[0], kernel[1]], dtype=np.int64),

            'input_feature_channel': 1,
            'output_feature_channel': 0,
            'kernel_spatial_idx': np.array([2, 3], dtype=np.int64),
            'reshape_kernel': True,
        }
        node_attrs.update(layout_attrs())

        # update the attributes of the node
        Convolution.update_node_stat(node, node_attrs)
        return __class__.enabled


class DeconvFrontExtractor(FrontExtractorOp):
    op = 'Deconvolution'
    enabled = True

    @staticmethod
    def get_pad(node, input_shape, kernel_shape):
        padding = np.add.reduce(node.pad, axis=1)
        padding[node.spatial_dims] = node.stride[node.spatial_dims] * (input_shape[node.spatial_dims] - 1) + 1 + \
                                     (kernel_shape[node.spatial_dims] - 1) * node.dilation[node.spatial_dims]
        padding[node.spatial_dims] = padding[node.spatial_dims] - node.output_spatial_shape;
        padding[node.spatial_dims] = (padding[node.spatial_dims] + 1) / 2
        return np.array([[0, 0], [0, 0], [padding[2], padding[2]], [padding[3], padding[3]]], dtype=np.int64)

    @staticmethod
    def extract(node):
        attr = get_mxnet_layer_attrs(node.symbol_dict)

        kernel = attr.tuple("kernel", int, None)
        stride = attr.tuple("stride", int, (1, 1))
        padding = attr.tuple("pad", int, (0, 0))
        dilate = attr.tuple("dilate", int, (1, 1))
        group = attr.int("num_group", 1)
        output = attr.int("num_filter", None)
        bias_term = attr.str("no_bias", 'True') == 'False'
        target_shape = attr.tuple("target_shape", int, None)
        if target_shape:
            target_shape = np.array(target_shape, dtype=np.int64)

        node_attrs = {
            'type': 'Deconvolution',
            'op': 'Deconv2D',
            'bias_addable': True,
            'bias_term': bias_term,
            'pad': np.array([[0, 0], [0, 0], [padding[0], padding[0]], [padding[1], padding[1]]], dtype=np.int64),
            'pad_spatial_shape': np.array([[padding[0], padding[0]], [padding[1], padding[1]]], dtype=np.int64),
            'dilation': np.array([1, 1, dilate[0], dilate[1]], dtype=np.int64),
            'output_spatial_shape': target_shape,
            'output_shape': None,
            'stride': np.array([1, 1, stride[0], stride[1]], dtype=np.int64),
            'group': group,
            'output': output,
            'kernel_spatial': np.array([kernel[0], kernel[1]], dtype=np.int64),
            'input_feature_channel': 0,
            'output_feature_channel': 1,
            'kernel_spatial_idx': np.array([2, 3], dtype=np.int64),
            'reshape_kernel': True,
            'get_pad': DeconvFrontExtractor.get_pad
        }
        node_attrs.update(layout_attrs())

        # update the attributes of the node
        Convolution.update_node_stat(node, node_attrs)
        return __class__.enabled
