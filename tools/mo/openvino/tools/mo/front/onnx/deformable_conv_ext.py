# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr, get_onnx_autopad
from openvino.tools.mo.ops.deformable_convolution import DeformableConvolution


class DeformableConvExtractor(FrontExtractorOp):
    op = 'DeformableConv2D'
    enabled = True

    @classmethod
    def extract(cls, node):
        # Extract pads attribute
        # In case if pads is not specified it will be set in default (1) in infer function
        pads = onnx_attr(node, 'pads', 'ints', default=None, dst_type=lambda x: int64_array(x))
        assert pads is None or len(pads) % 2 == 0
        final_pad = None
        if pads is not None:
            pads = pads.reshape([2, -1])
            pads = np.transpose(pads)
            final_pad = int64_array([[0, 0], [0, 0], *pads])

        # Extract dilations attribute
        # In case if dilations is not specified it will be set in default (1) in infer function
        dilations = onnx_attr(node, 'dilations', 'ints', default=None, dst_type=lambda x: int64_array(x))
        final_dilations = int64_array([1, 1, *dilations]) if dilations is not None else None

        # Extract dilations attribute
        # In case if dilations is not specified it will be set in default (1) in infer function
        strides = onnx_attr(node, 'strides', 'ints', default=None, dst_type=lambda x: int64_array(x))
        final_strides = int64_array([1, 1, *strides]) if strides is not None else None

        kernel_shape = onnx_attr(node, 'kernel_shape', 'ints', default=None)
        auto_pad = onnx_attr(node, 'auto_pad', 's', default=None, dst_type=get_onnx_autopad)
        group = onnx_attr(node, 'group', 'i', default=1, dst_type=lambda x: int64_array(x))
        deformable_groups = onnx_attr(node, 'deformable_groups', 'i', default=1)

        attrs = {
            'op': __class__.op,
            'auto_pad': auto_pad,
            'bias_addable': False,
            'bias_term': False,
            'pad': final_pad,
            'pad_spatial_shape': int64_array(pads) if pads is not None else None,
            'dilation': final_dilations,
            'output_spatial_shape': None,
            'output_shape': None,
            'stride': final_strides,
            'group': group,
            'deformable_group': deformable_groups,
            'output': None,
            'weights_index': 2,
            'kernel_spatial': int64_array(kernel_shape) if kernel_shape is not None else None,

            'input_feature_channel': 1,
            'output_feature_channel': 0,
            'kernel_spatial_idx': None,  # Will be calculated in infer function (np.array([2, 3]))

            'spatial_dims': None,  # Will be calculated in infer function
            'channel_dims': int64_array([1]),
            'batch_dims': int64_array([0]),
            'layout': 'NCHW'
        }

        # update the attributes of the node
        DeformableConvolution.update_node_stat(node, attrs)
        return cls.enabled
