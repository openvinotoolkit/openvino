# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.front.extractor import FrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.ops.deformable_convolution import DeformableConvolution


class ModulatedDeformableConvolutionExtractor(FrontExtractorOp):
    op = '_contrib_ModulatedDeformableConvolution'
    enabled = True

    @classmethod
    def extract(cls, node):
        attr = get_mxnet_layer_attrs(node.symbol_dict)

        kernel = attr.tuple("kernel", int, None)
        stride = attr.tuple("stride", int, tuple(np.ones(len(kernel), dtype=np.int64)))
        padding = attr.tuple("pad", int, tuple(np.zeros(len(kernel), dtype=np.int64)))
        dilate = attr.tuple("dilate", int, tuple(np.ones(len(kernel), dtype=np.int64)))
        num_deformable_group = attr.int("num_deformable_group", 1)
        num_group = attr.int("num_group", 1)
        output = attr.int("num_filter", None)
        bias_term = attr.str("no_bias", 'False') == 'False'

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
            'group': num_group,
            'deformable_group': num_deformable_group,
            'output': output,
            'kernel_spatial': np.array([k for k in kernel], dtype=np.int64),
            'use_bilinear_interpolation_padding': True,

            'input_feature_channel': 1,
            'output_feature_channel': 0,
            'kernel_spatial_idx': None,
            'reshape_kernel': True,
            'weights_index': 2,
            'in_ports_count': 4,

            'spatial_dims': None,
            'channel_dims': np.array([1], dtype=np.int64),
            'batch_dims': np.array([0], dtype=np.int64),
            'layout': 'NCHW',
        }

        # update the attributes of the node
        DeformableConvolution.update_node_stat(node, node_attrs)
        return cls.enabled
