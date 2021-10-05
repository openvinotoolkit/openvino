# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.front.extractor import FrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.ops.convolution import Convolution


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

        final_dilations = np.array([1, 1, *[d for d in dilate]], dtype=np.int64) if dilate is not None else None

        node_attrs = {
            'op': __class__.op,
            'bias_addable': True,
            'bias_term': bias_term,
            'pad': np.array([[0, 0], [0, 0], *[[pad, pad] for pad in padding]], dtype=np.int64),
            'pad_spatial_shape': np.array([[pad, pad] for pad in padding], dtype=np.int64),
            'dilation': final_dilations,
            'output_spatial_shape': None,
            'output_shape': None,
            'stride': np.array([1, 1, *[s for s in stride]], dtype=np.int64),
            'group': group,
            'output': output,
            'kernel_spatial': np.array([k for k in kernel], dtype=np.int64),

            'input_feature_channel': 1,
            'output_feature_channel': 0,
            'kernel_spatial_idx': None,
            'reshape_kernel': True,

            'spatial_dims': None,
            'channel_dims': np.array([1], dtype=np.int64),
            'batch_dims': np.array([0], dtype=np.int64),
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
        padding[node.spatial_dims] = padding[node.spatial_dims] - node.output_spatial_shape;
        padding[node.spatial_dims] = (padding[node.spatial_dims] + 1) / 2
        return np.array([[0, 0], [0, 0], *[[pad, pad] for pad in padding[2:]]], dtype=np.int64)

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
            target_shape = np.array(target_shape, dtype=np.int64)

        final_dilations = np.array([1, 1, *[d for d in dilate]], dtype=np.int64) if dilate is not None else None
        node_attrs = {
            'op': __class__.op,
            'type': 'Deconvolution',
            'bias_addable': True,
            'bias_term': bias_term,
            'pad': np.array([[0, 0], [0, 0], *[[pad, pad] for pad in padding]], dtype=np.int64),
            'pad_spatial_shape': np.array([[pad, pad] for pad in padding], dtype=np.int64),
            'dilation': final_dilations,
            'output_spatial_shape': target_shape,
            'original_output_spatial_shape': target_shape,
            'output_shape': None,
            'stride': np.array([1, 1, *[s for s in stride]], dtype=np.int64),
            'group': group,
            'output': output,
            'kernel_spatial': np.array([k for k in kernel], dtype=np.int64),
            'input_feature_channel': 1,
            'output_feature_channel': 0,
            'kernel_spatial_idx': None,
            'reshape_kernel': True,

            'spatial_dims': None,
            'channel_dims': np.array([1], dtype=np.int64),
            'batch_dims': np.array([0], dtype=np.int64),
            'layout': 'NCHW',
            'get_pad': DeconvFrontExtractor.get_pad,
        }

        output_padding = attr.tuple("adj", int, None)
        if target_shape is None and output_padding:
            node_attrs["output_padding"] = np.array([0, 0, *[s for s in output_padding]], dtype=np.int64)

        # update the attributes of the node
        Convolution.update_node_stat(node, node_attrs)
        return cls.enabled
