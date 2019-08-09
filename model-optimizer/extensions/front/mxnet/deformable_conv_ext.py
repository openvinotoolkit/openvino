"""
 Copyright (c) 2019 Intel Corporation

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

from mo.front.extractor import FrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.ops.deformable_convolution import DeformableConvolution


class DeformableConvolutionExtractor(FrontExtractorOp):
    op = '_contrib_DeformableConvolution'
    enabled = True

    @staticmethod
    def extract(node):
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
        DeformableConvolution.update_node_stat(node, node_attrs)
        return __class__.enabled
