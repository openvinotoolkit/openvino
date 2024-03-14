# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math

from openvino.tools.mo.front.mxnet.conv_ext import DeconvFrontExtractor
from openvino.tools.mo.ops.interpolate import Interpolate
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from openvino.tools.mo.ops.convolution import Convolution


class UpSamplingFrontExtractor(FrontExtractorOp):
    op = 'UpSampling'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        scale = attrs.int("scale", 1)
        num_filter = attrs.int("num_filter", 0)
        mode = attrs.str("sample_type", None)
        if mode == 'nearest':
            node_attrs = {
                'factor': attrs.int("scale", 1),
                'mode': mode,
                'antialias': 0,
                'axes': int64_array([2, 3]),
            }
            Interpolate.update_node_stat(node, node_attrs)
        elif mode == 'bilinear':
            """
            Bilinear UpSampling uses deconvolution algorithm under the hood.
            For MXNet Bilinear UpSampling op just wrapper over Deconvolution op.
            Inputs data:
                input1 - input data
                input2 - deconvolution weight
            """
            kernel = 2 * scale - scale % 2
            stride = scale
            pad = math.ceil((scale - 1) / 2)
            num_group = num_filter

            node_attrs = {
                'op': __class__.op,
                'type': 'Deconvolution',
                'bias_addable': True,
                'bias_term':  False,
                'pad': int64_array([[0, 0], [0, 0], [pad, pad], [pad, pad]]),
                'pad_spatial_shape': int64_array([[pad, pad], [pad, pad]]),
                'dilation': None,
                'output_spatial_shape': None,
                'output_shape': None,
                'stride': int64_array([1, 1, stride, stride]),
                'group': num_group,
                'output': num_filter,
                'kernel_spatial': int64_array([kernel, kernel]),
                'input_feature_channel': 0,
                'output_feature_channel': 1,
                'kernel_spatial_idx': None,
                'reshape_kernel': True,
                'spatial_dims': None,
                'channel_dims': int64_array([1]),
                'batch_dims': int64_array([0]),
                'layout': 'NCHW',
                'get_pad': DeconvFrontExtractor.get_pad,
            }
            Convolution.update_node_stat(node, node_attrs)
        return cls.enabled
