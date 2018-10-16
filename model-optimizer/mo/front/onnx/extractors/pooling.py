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
from mo.front.common.partial_infer.pooling import pool_explicit_padding_infer
from mo.front.onnx.extractors.utils import onnx_attr
from mo.utils.error import Error
import logging as log


def onnx_pooling_extractor(node):
    pads = np.array(onnx_attr(node, 'pads', 'ints', default=[0,0,0,0]), dtype=np.int64)

    # Try to convert slightly incorrect models with insufficient pad parameters
    if pads.size == 2:
        log.warning('Node {} has pad = {} which is ill-formed -- it should consist of 4 elements.'.format(node.name, pads))
        pads = np.concatenate([pads, pads])
        log.warning('Extended pads to {}'.format(pads))

    assert len(pads)%2 == 0
    pads = pads.reshape([2,-1])
    pads = np.transpose(pads)
    strides = np.array(onnx_attr(node, 'strides', 'ints', default=[1,1]), dtype=np.int64)
    kernel_shape = np.array(onnx_attr(node, 'kernel_shape', 'ints'), dtype=np.int64)
    # exclude_pad = True only when count_include_pad == 0
    exclude_pad = onnx_attr(node, 'count_include_pad', 'i', default=0) == 0
    global_pooling = 0
    if node.op == 'MaxPool':
        method = 'max'
    elif node.op == 'AveragePool':
        method = 'avg'
    else:
        raise Error('Unsupported pooling op {}', node.op)
    
    # TODO check if it is a correct choice for ONNX
    pooling_convention = 'valid'  # for Caffe rounding type should be ceil
    rt = 'floor'

    auto_pad = onnx_attr(node, 'auto_pad', 's', default=None, dst_type=lambda x: x.decode().lower())

    attrs = {
        'auto_pad': auto_pad,
        'type': 'Pooling',
        'window': np.array([1, 1, kernel_shape[0], kernel_shape[1]], dtype=np.int64),
        'stride': np.array([1, 1, strides[0], strides[1]], dtype=np.int64),
        'pad': np.array([[0, 0], [0, 0], pads[0], pads[1]], dtype=np.int64),
        'pad_spatial_shape': np.array([pads[0], pads[1]], dtype=np.int64),
        'pool_method': method,
        'exclude_pad': 'true' if exclude_pad else 'false',
        'infer': pool_explicit_padding_infer,
        'global_pool': global_pooling,
        'output_spatial_shape': None,
        'rounding_type': rt
    }

    attrs.update(layout_attrs())
    attrs['pooling_convention'] = pooling_convention
    return attrs
