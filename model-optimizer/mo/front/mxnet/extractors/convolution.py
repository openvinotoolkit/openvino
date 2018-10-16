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
from mo.front.common.partial_infer.convolution import mxnet_conv2d_infer


def convolution_ext(attr):
    kernel = attr.tuple("kernel", int, None)
    stride = attr.tuple("stride", int, (1, 1))
    padding = attr.tuple("pad", int, (0, 0))
    dilate = attr.tuple("dilate", int, (1, 1))
    group = attr.int("num_group", 1)
    output = attr.int("num_filter", None)

    node_attrs = {
        'op': 'Conv2D',
        'bias_addable': True,
        'bias_term': False,
        'pad': np.array([[0, 0], [0, 0], [padding[0], padding[0]], [padding[1], padding[1]]], dtype=np.int64),
        'pad_spatial_shape': np.array([[padding[0], padding[0]], [padding[1], padding[1]]], dtype=np.int64),
        'dilation': np.array([1, 1, dilate[0], dilate[1]], dtype=np.int64),
        'output_spatial_shape': None,
        'output_shape': None,
        'kernel_spatial': np.array([kernel[0], kernel[1]], dtype=np.int64),
        'stride': np.array([1, 1, stride[0], stride[1]], dtype=np.int64),
        'type': 'Convolution',
        'group': group,
        'output': output,
        'layout': 'NCHW',
        'infer': mxnet_conv2d_infer
    }
    node_attrs.update(layout_attrs())
    return node_attrs
