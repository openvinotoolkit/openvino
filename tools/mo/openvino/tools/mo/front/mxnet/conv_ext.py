# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from openvino.tools.mo.ops.convolution import Convolution


class ConvFrontExtractor(FrontExtractorOp):
    op = 'Convolution'
    enabled = True

    @classmethod
    def extract(cls, node):
        attr = get_mxnet_layer_attrs(node.symbol_dict)

        kernel = attr.tuple("kernel", int, None)
        stride = attr.tuple("stride", int, tuple(np.ones(len(kernel), dtype=np.int64)))
        padding = attr.tuple("pad", int, tuple(np.zeros(len(kernel), dtype=np.int64)))
        dilate = attr.tuple("dilate", int, tuple(np.ones(len(kernel), dtype=np.int64)))
        group = attr.int("num_group", 1)
        output = attr.int("num_filter", None)
        bias_term = not attr.bool("no_bias", False)

        final_dilations = int64_array([1, 1, *[d for d in dilate]]) if dilate is not None else None

        node_attrs = {
            'op': __class__.op,
            'bias_addable': True,
            'bias_term': bias_term,
            'pad': int64_array([[0, 0], [0, 0], *[[pad, pad] for pad in padding]]),
            'pad_spatial_shape': int64_array([[pad, pad] for pad in padding]),
            'dilation': final_dilations,
            'output_spatial_shape': None,
            'output_shape': None,
            'stride': int64_array([1, 1, *[s for s in stride]]),
            'group': group,
            'output': output,
            'kernel_spatial': int64_array([k for k in kernel]),

            'input_feature_channel': 1,
            'output_feature_channel': 0,
            'kernel_spatial_idx': None,
            'reshape_kernel': True,

            'spatial_dims': None,
            'channel_dims': int64_array([1]),
            'batch_dims': int64_array([0]),
            'layout': 'NCHW',
        }

        # update the attributes of the node
        Convolution.update_node_stat(node, node_attrs)
        return cls.enabled


class DeconvFrontExtractor(FrontExtractorOp):
    op = 'Deconvolution'
    enabled = True

    @staticmethod
    def get_pad(node, input_shape, kernel_shape):
        padding = np.add.reduce(node.pad, axis=1)
        padding[node.spatial_dims] = node.stride[node.spatial_dims] * (input_shape[node.spatial_dims] - 1) + 1 + \
                                     (kernel_shape[node.spatial_dims] - 1) * node.dilation[node.spatial_dims]
        padding[node.spatial_dims] = padding[node.spatial_dims] - node.output_spatial_shape
        padding[node.spatial_dims] = (padding[node.spatial_dims] + 1) / 2
        return int64_array([[0, 0], [0, 0], *[[pad, pad] for pad in padding[2:]]])

    @classmethod
    def extract(cls, node):
        attr = get_mxnet_layer_attrs(node.symbol_dict)

        kernel = attr.tuple("kernel", int, None)
        stride = attr.tuple("stride", int, tuple(np.ones(len(kernel), dtype=np.int64)))
        padding = attr.tuple("pad", int, tuple(np.zeros(len(kernel), dtype=np.int64)))
        dilate = attr.tuple("dilate", int, tuple(np.ones(len(kernel), dtype=np.int64)))
        group = attr.int("num_group", 1)
        output = attr.int("num_filter", None)
        bias_term = not attr.bool("no_bias", True)
        target_shape = attr.tuple("target_shape", int, None)
        if target_shape:
            target_shape = int64_array(target_shape)

        final_dilations = int64_array([1, 1, *[d for d in dilate]]) if dilate is not None else None
        node_attrs = {
            'op': __class__.op,
            'type': 'Deconvolution',
            'bias_addable': True,
            'bias_term': bias_term,
            'pad': int64_array([[0, 0], [0, 0], *[[pad, pad] for pad in padding]]),
            'pad_spatial_shape': int64_array([[pad, pad] for pad in padding]),
            'dilation': final_dilations,
            'output_spatial_shape': target_shape,
            'original_output_spatial_shape': target_shape,
            'output_shape': None,
            'stride': int64_array([1, 1, *[s for s in stride]]),
            'group': group,
            'output': output,
            'kernel_spatial': int64_array([k for k in kernel]),
            'input_feature_channel': 1,
            'output_feature_channel': 0,
            'kernel_spatial_idx': None,
            'reshape_kernel': True,

            'spatial_dims': None,
            'channel_dims': int64_array([1]),
            'batch_dims': int64_array([0]),
            'layout': 'NCHW',
            'get_pad': DeconvFrontExtractor.get_pad,
        }

        output_padding = attr.tuple("adj", int, None)
        if target_shape is None and output_padding:
            node_attrs["output_padding"] = int64_array([0, 0, *[s for s in output_padding]])

        # update the attributes of the node
        Convolution.update_node_stat(node, node_attrs)
        return cls.enabled
