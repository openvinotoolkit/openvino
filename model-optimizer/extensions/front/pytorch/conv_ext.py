"""
 Copyright (C) 2018-2020 Intel Corporation

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

from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import FrontExtractorOp
from mo.ops.convolution import Convolution
from mo.utils.error import Error


class Conv2dFrontExtractor(FrontExtractorOp):
    op = 'Conv2d'
    enabled = True

    @classmethod
    def extract(cls, node):
        # Extract pads attribute
        pads = np.array(node.module.padding, dtype=np.int64).reshape(1, 2)
        pads = np.repeat(pads, 2, axis=0)
        final_pads = np.array([[0, 0], [0, 0], *pads], dtype=np.int64)

        # Extract strides attribute
        strides = node.module.stride
        final_strides = np.array([1, 1, *strides], dtype=np.int64)

        # Extract strides attribute
        dilations = node.module.dilation
        final_dilations = np.array([1, 1, *dilations], dtype=np.int64)

        attrs = {
            'op': __class__.op,
            'pad': final_pads,
            'stride': final_strides,
            'dilation': final_dilations,
            'group': 1,
            'kernel_spatial': np.array(node.module.kernel_size, dtype=np.int64),

            'input_feature_channel': 1,
            'output_feature_channel': 0,

            'channel_dims': np.array([1], dtype=np.int64),
            'batch_dims': np.array([0], dtype=np.int64),
            'layout': 'NCHW',
        }

        # update the attributes of the node
        Convolution.update_node_stat(node, attrs)
        return cls.enabled
