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

import numpy as np

from mo.front.common.extractors.utils import layout_attrs
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr, get_onnx_autopad
from mo.ops.convolution import Convolution
from mo.utils.error import Error
from mo.front.common.partial_infer.utils import int64_array


class ConvFrontExtractor(FrontExtractorOp):
    op = 'Conv'
    enabled = True

    @staticmethod
    def extract(node):
        # Extract pads attribute
        # In case if pads is not specified it will be set in default (1) in infer function
        pads = onnx_attr(node, 'pads', 'ints', default=None, dst_type=lambda x: np.array(x, dtype=np.int64))
        assert pads is None or len(pads) % 2 == 0
        final_pad = None
        if pads is not None:
            pads = pads.reshape([2, -1])
            pads = np.transpose(pads)
            final_pad = np.array([[0, 0], [0, 0], *pads], dtype=np.int64)

        # Extract dilations attribute
        # In case if dilations is not specified it will be set in default (1) in infer function
        dilations = onnx_attr(node, 'dilations', 'ints', default=None, dst_type=lambda x: np.array(x, dtype=np.int64))
        final_dilations = np.array([1, 1, *dilations], dtype=np.int64) if dilations is not None else None

        # Extract dilations attribute
        # In case if dilations is not specified it will be set in default (1) in infer function
        strides = onnx_attr(node, 'strides', 'ints', default=None, dst_type=lambda x: np.array(x, dtype=np.int64))
        final_strides = np.array([1, 1, *strides], dtype=np.int64) if strides is not None else None

        kernel_shape = onnx_attr(node, 'kernel_shape', 'ints', default=None)
        auto_pad = onnx_attr(node, 'auto_pad', 's', default=None, dst_type=get_onnx_autopad)
        group = onnx_attr(node, 'group', 'i', default=1, dst_type=lambda x: np.array(x, dtype=np.int64))

        attrs = {
            'op': __class__.op,
            'auto_pad': auto_pad,
            'bias_addable': True,
            'bias_term': None,
            'pad': final_pad,
            'pad_spatial_shape': np.array(pads, dtype=np.int64) if pads is not None else None,
            'dilation': final_dilations,
            'output_spatial_shape': None,
            'output_shape': None,
            'stride': final_strides,
            'group': group,
            'output': None,
            'kernel_spatial': np.array(kernel_shape, dtype=np.int64) if kernel_shape is not None else None,

            'input_feature_channel': 1,
            'output_feature_channel': 0,
            'kernel_spatial_idx': None,  # Will be calculated in infer function (np.array([2, 3]))

            'spatial_dims': None,  # Will be calculated in infer function
            'channel_dims': np.array([1], dtype=np.int64),
            'batch_dims': np.array([0], dtype=np.int64),
            'layout': 'NCHW'
        }

        # update the attributes of the node
        Convolution.update_node_stat(node, attrs)
        return __class__.enabled


class ConvTransposeFrontExtractor(FrontExtractorOp):
    op = 'ConvTranspose'
    enabled = True

    @staticmethod
    def get_pad(node, input_shape, kernel_shape):
        # Reference: https://github.com/onnx/onnx/blob/master/docs/Operators.md#ConvTranspose
        input_shape = node.in_node(0).shape
        pad = np.zeros((len(input_shape), 2), dtype=np.int64)
        total_padding = int64_array([node.stride[node.spatial_dims][x] *
                                     (input_shape[node.spatial_dims][x] - 1) +
                                     node.output_padding[node.spatial_dims][x] +
                                     kernel_shape[node.kernel_spatial_idx][x] -
                                     node.output_spatial_shape[x] for x in range(len(node.spatial_dims))])
        if node.has_valid('auto_pad') and node.auto_pad != 'same_upper':
            pad[node.spatial_dims] = int64_array(
                [[total_padding[x] / 2, total_padding[x] - (total_padding[x] // 2)] for x in
                 range(len(node.spatial_dims))])
        else:
            pad[node.spatial_dims] = int64_array(
                [[total_padding[x] - (total_padding[x] // 2), total_padding[x] / 2] for x in
                 range(len(node.spatial_dims))])
        return pad

    @staticmethod
    def extract(node):
        pads = onnx_attr(node, 'pads', 'ints', dst_type=int64_array)
        auto_pad = onnx_attr(node, 'auto_pad', 's', default=None, dst_type=get_onnx_autopad)

        if pads is not None:
            if len(pads) % 2 != 0:
                raise Error(
                    'ConvTranspose node {} specifies pads = {} which has odd number of elements. The model is not correct.',
                    node.soft_get('name'),
                    pads
                )
            pads = pads.reshape([2, -1])
            pads = np.transpose(pads)

        final_pads = int64_array([[0, 0], [0, 0], *pads]) if pads is not None else None

        dilations = onnx_attr(node, 'dilations', 'ints', default=None)
        final_dilations = int64_array([1, 1, *dilations]) if dilations is not None else None

        strides = onnx_attr(node, 'strides', 'ints', default=None)
        final_strides = int64_array([1, 1, *strides]) if strides is not None else None

        kernel_shape = onnx_attr(node, 'kernel_shape', 'ints', dst_type=int64_array)

        if kernel_shape is None:
            raise Error(
                'ConvTranspose node {} doesn\'t have explicitly defined kernel_shape. It is not supported.',
                node.soft_get('name')
            )

        output_padding = onnx_attr(node, 'output_padding', 'ints', default=None)
        final_output_padding = int64_array([0, 0, *output_padding]) if output_padding is not None else None

        output_shape = onnx_attr(node, 'output_shape', 'ints', default=None, dst_type=int64_array)

        attrs = {
            'type': 'Deconvolution',
            'op': 'Deconv2D',
            'auto_pad': auto_pad,
            'bias_addable': True,
            'bias_term': None,  # will be deduced later; not really needed
            'pad': final_pads,
            'dilation': final_dilations,
            'output_spatial_shape': output_shape,
            'output_shape': None,
            'output_padding': final_output_padding,
            'stride': final_strides,
            'group': onnx_attr(node, 'group', 'i', default=1),
            'output': None,

            'spatial_dims': None,  # Will be calculated in infer function
            'channel_dims': int64_array([1]),
            'batch_dims': int64_array([0]),
            'layout': 'NCHW',

            'input_feature_channel': 0,
            'output_feature_channel': 1,
            'get_pad': ConvTransposeFrontExtractor.get_pad,
            'get_output_feature_dim': lambda node: node.kernel_shape[node.output_feature_channel] * node.group,
        }

        # update the attributes of the node
        Convolution.update_node_stat(node, attrs)
        return __class__.enabled
