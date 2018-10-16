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

from mo.front.onnx.extractors.utils import onnx_attr
from mo.front.common.extractors.utils import layout_attrs
from mo.front.common.partial_infer.convolution import onnx_conv2d_infer
from mo.utils.error import Error


def onnx_conv_extractor(node):

    pads = np.array(onnx_attr(node, 'pads', 'ints', default=[0,0,0,0]), dtype=np.int64)
    assert len(pads)%2 == 0
    pads = pads.reshape([2,-1])
    pads = np.transpose(pads)
    auto_pad = onnx_attr(node, 'auto_pad', 's', default=None, dst_type= lambda x: x.decode().lower())
    dilations = np.array(onnx_attr(node, 'dilations', 'ints', default=[1,1]), dtype=np.int64)
    strides = np.array(onnx_attr(node, 'strides', 'ints', default=[1,1]), dtype=np.int64)
    kernel_shape = onnx_attr(node, 'kernel_shape', 'ints')
    attrs = {
        'type': 'Convolution',
        'op': 'Conv2D',
        'auto_pad': auto_pad,
        'bias_addable': True,
        'bias_term': node.in_nodes() == 3,
        'pad': np.array([[0, 0], [0, 0], pads[0], pads[1]], dtype=np.int64),
        'pad_spatial_shape': np.array([pads[0], pads[1]], dtype=np.int64),
        'dilation': np.array([1, 1, dilations[0], dilations[1]], dtype=np.int64),
        'output_spatial_shape': None,
        'output_shape': None,
        'stride': np.array([1, 1, strides[0], strides[1]], dtype=np.int64),
        'group': onnx_attr(node, 'group', 'i', default=1),
        'output': None,
        'kernel_spatial': np.array([kernel_shape[0], kernel_shape[1]], dtype=np.int64),  # TODO WARNING Don't misuse X/Y
        'input_feature_channel': None,
        'output_feature_channel': None,
        'infer': onnx_conv2d_infer,
    }
    attrs.update(layout_attrs())
    return attrs


# TODO: move it outside as an extractor class when convolution/deconvolution code is consolidated as dedicated ops
def onnx_conv_trans_extractor(node):

    int64array = lambda x: np.array(x, dtype=np.int64)

    pads = onnx_attr(node, 'pads', 'ints', dst_type=int64array)

    if pads is None:
        if onnx_attr(node, 'auto_pad', 's') is not None:
            raise Error(
                'ConvTranspose node {} doesn\'t define pads explicitly but uses auto_pad. It is not supported.',
                node.soft_get('name')
            )
        else:
            pads = np.array([0,0,0,0], dtype=np.int64)

    if len(pads)%2 != 0:
        raise Error(
            'ConvTranspose node {} specifies pads = {} which has odd number of elements. The model is not correct.',
            node.soft_get('name'),
            pads
        )

    pads = pads.reshape([2,-1])
    pads = np.transpose(pads)
    dilations = int64array(onnx_attr(node, 'dilations', 'ints', default=[1,1]))
    strides = int64array(onnx_attr(node, 'strides', 'ints', default=[1,1]))
    kernel_shape = onnx_attr(node, 'kernel_shape', 'ints', dst_type=int64array)

    if kernel_shape is None:
        raise Error(
            'ConvTranspose node {} doesn\'t have explicitly defined kernel_shape. It is not supported.',
            node.soft_get('name')
        )

    output_padding = onnx_attr(node, 'output_padding', 'ints', default=[0,0])

    output_shape = onnx_attr(node, 'output_shape', 'ints')
    if output_shape is not None:
        raise Error(
            'ConvTranspose node {} explicitly specifies output_shape. It is not supported.',
            node.soft_get('name')
        )

    attrs = {
        'type': 'Deconvolution',
        'op': 'Deconv2D',
        'bias_addable': True,
        'bias_term': None, # will be deduced later; not really needed
        'pad': int64array([[0, 0], [0, 0], pads[0], pads[1]]),
        'pad_spatial_shape': int64array([pads[0], pads[1]]),
        'dilation': int64array([1, 1, dilations[0], dilations[1]]),
        'output_spatial_shape': None,
        'output_shape': None,
        'output_padding': int64array([output_padding[0], output_padding[1]]),
        'stride': int64array([1, 1, strides[0], strides[1]]),
        'infer': onnx_conv2d_infer,
        'group': onnx_attr(node, 'group', 'i', default=1),
        'output': None,
        'spatial_dims': int64array([2,3]),
        'channel_dims': int64array([1]),
        'batch_dims': int64array([0]),
        'kernel_spatial': int64array([kernel_shape[0], kernel_shape[1]])  # TODO WARNING Don't misuse X/Y
    }
    attrs.update(layout_attrs())
    return attrs


