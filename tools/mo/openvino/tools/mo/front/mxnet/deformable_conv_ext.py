# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from openvino.tools.mo.ops.deformable_convolution import DeformableConvolution


class DeformableConvolutionExtractor(FrontExtractorOp):
    op = '_contrib_DeformableConvolution'
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
            'group': num_group,
            'deformable_group': num_deformable_group,
            'output': output,
            'kernel_spatial': int64_array([k for k in kernel]),

            'input_feature_channel': 1,
            'output_feature_channel': 0,
            'kernel_spatial_idx': None,
            'reshape_kernel': True,
            'weights_index': 2,

            'spatial_dims': None,
            'channel_dims': int64_array([1]),
            'batch_dims': int64_array([0]),
            'layout': 'NCHW',
        }

        # update the attributes of the node
        DeformableConvolution.update_node_stat(node, node_attrs)
        return cls.enabled
