# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr, get_onnx_autopad
from openvino.tools.mo.ops.pooling import Pooling
from openvino.tools.mo.utils.error import Error


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
    kernel_shape = onnx_attr(node, 'kernel_shape', 'ints', default=None, dst_type=lambda x: int64_array(x))
    final_kernel_shape = int64_array([1, 1, *[x for x in kernel_shape]]) if kernel_shape is not None else None

    pads = onnx_attr(node, 'pads', 'ints', default=None, dst_type=lambda x: int64_array(x))

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
        final_pads = int64_array([[0, 0], [0, 0], *[p for p in pads]])

    # Extract strides attribute
    # In case if strides is not specified it will be set in default (1) in infer function
    strides = onnx_attr(node, 'strides', 'ints', default=None, dst_type=lambda x: int64_array(x))
    final_strides = int64_array([1, 1, *[x for x in strides]]) if strides is not None else None

    dilation = onnx_attr(node, 'dilations', 'ints', default=None, dst_type=lambda x: int64_array(x))
    final_dilation = int64_array([1, 1, *[x for x in dilation]]) if dilation is not None else None

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
        'pad_spatial_shape': int64_array(pads) if pads is not None else None,
        'pool_method': method,
        'exclude_pad': True if exclude_pad else False,
        'global_pool': global_pooling,
        'output_spatial_shape': None,
        'rounding_type': rt,
        'dilation': final_dilation,

        'spatial_dims': None,
        'channel_dims': int64_array([1]),
        'batch_dims': int64_array([0]),
        'layout': 'NCHW',

        'pooling_convention': pooling_convention
    }
    return attrs
