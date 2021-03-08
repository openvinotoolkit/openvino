"""
 Copyright (C) 2018-2021 Intel Corporation

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

import logging as log

import numpy as np

from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr, get_onnx_autopad
from mo.ops.pooling import Pooling
from mo.utils.error import Error


class AveragePoolFrontExtractor(FrontExtractorOp):
    op = 'AveragePool'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = common_onnx_pool_extractor(node)

        Pooling.update_node_stat(node, attrs)
        return cls.enabled


class MaxPoolFrontExtractor(FrontExtractorOp):
    op = 'MaxPool'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = common_onnx_pool_extractor(node)

        Pooling.update_node_stat(node, attrs)
        return cls.enabled


class GlobalAveragePoolFrontExtractor(FrontExtractorOp):
    op = 'GlobalAveragePool'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = common_onnx_pool_extractor(node)
        attrs.update({'pooling_convention': 'full',
                      'global_pool': True,
                     })

        Pooling.update_node_stat(node, attrs)
        return cls.enabled


class GlobalMaxPoolFrontExtractor(FrontExtractorOp):
    op = 'GlobalMaxPool'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = common_onnx_pool_extractor(node)
        attrs.update({'pooling_convention': 'full',
                      'global_pool': True,
                     })

        Pooling.update_node_stat(node, attrs)
        return cls.enabled


def common_onnx_pool_extractor(node):
    kernel_shape = onnx_attr(node, 'kernel_shape', 'ints', default=None, dst_type=lambda x: np.array(x, dtype=np.int64))
    final_kernel_shape = np.array([1, 1, *[x for x in kernel_shape]], dtype=np.int64) if kernel_shape is not None else None

    pads = onnx_attr(node, 'pads', 'ints', default=None, dst_type=lambda x: np.array(x, dtype=np.int64))

    if kernel_shape is not None and pads is not None and kernel_shape.size * 2 != pads.size:
        log.warning('Node {} has pad = {} which is ill-formed -- it should have even amount of elements.'.format(
            node.soft_get('name', node.id), pads))

        # Try to convert slightly incorrect models with insufficient pad parameters
        assert pads.size == kernel_shape.size
        pads = np.concatenate([pads, pads])
        log.warning('Extended pads to {}'.format(pads))

    final_pads = None
    if pads is not None:
        assert len(pads) % 2 == 0
        pads = pads.reshape([2, -1])
        pads = np.transpose(pads)
        final_pads = np.array([[0, 0], [0, 0], *[p for p in pads]], dtype=np.int64)

    # Extract strides attribute
    # In case if strides is not specified it will be set in default (1) in infer function
    strides = onnx_attr(node, 'strides', 'ints', default=None, dst_type=lambda x: np.array(x, dtype=np.int64))
    final_strides = np.array([1, 1, *[x for x in strides]], dtype=np.int64) if strides is not None else None

    dilations = onnx_attr(node, 'dilations', 'ints', default=None, dst_type=lambda x: np.array(x, dtype=np.int64))
    assert dilations is None or np.all(dilations == 1),\
        'Node {} has "dilations" attribute with values not equal to 1s which is not supported'.format(node.id)

    # exclude_pad = True only when count_include_pad == 0
    exclude_pad = onnx_attr(node, 'count_include_pad', 'i', default=0) == 0

    global_pooling = False
    if node.op in ['MaxPool', 'GlobalMaxPool']:
        method = 'max'
    elif node.op in ['AveragePool', 'GlobalAveragePool']:
        method = 'avg'
    else:
        raise Error('Unsupported pooling op {}', node.op)

    # TODO check if it is a correct choice for ONNX
    pooling_convention = 'valid'  # for Caffe rounding type should be ceil
    rt = 'floor' if onnx_attr(node, 'ceil_mode', 'i', default=0) == 0 else 'ceil'

    auto_pad = onnx_attr(node, 'auto_pad', 's', default=None, dst_type=get_onnx_autopad)
    if auto_pad:
        rt = 'ceil'

    attrs = {
        'op': node.op,
        'auto_pad': auto_pad,
        'window': final_kernel_shape,
        'stride': final_strides,
        'pad': final_pads,
        'pad_spatial_shape': np.array(pads, dtype=np.int64) if pads is not None else None,
        'pool_method': method,
        'exclude_pad': True if exclude_pad else False,
        'global_pool': global_pooling,
        'output_spatial_shape': None,
        'rounding_type': rt,

        'spatial_dims': None,
        'channel_dims': np.array([1], dtype=np.int64),
        'batch_dims': np.array([0], dtype=np.int64),
        'layout': 'NCHW',

        'pooling_convention': pooling_convention
    }
    return attrs
