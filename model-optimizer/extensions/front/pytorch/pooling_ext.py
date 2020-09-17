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
from mo.front.onnx.extractors.utils import onnx_attr, get_onnx_autopad
from mo.ops.pooling import Pooling
from extensions.ops.identity import Identity
from mo.utils.error import Error


class MaxPool2dFrontExtractor(FrontExtractorOp):
    op = 'MaxPool2d'
    enabled = True

    @classmethod
    def extract(cls, node):
        # Extract pads attribute
        pads = np.array([node.module.padding, node.module.padding], dtype=np.int64).reshape(1, 2)
        pads = np.repeat(pads, 2, axis=0)
        final_pads = np.array([[0, 0], [0, 0], *pads], dtype=np.int64)

        # Extract strides attribute
        strides = [node.module.stride, node.module.stride]
        final_strides = np.array([1, 1, *strides], dtype=np.int64)

        kernel_shape = [node.module.kernel_size, node.module.kernel_size]
        final_kernel_shape = np.array([1, 1, *kernel_shape], dtype=np.int64)

        attrs = {
            'op': node.op,
            'window': final_kernel_shape,
            'stride': final_strides,
            'pad': final_pads,
            'pool_method': 'max',

            'channel_dims': np.array([1], dtype=np.int64),
            'batch_dims': np.array([0], dtype=np.int64),
            'layout': 'NCHW',
        }

        # update the attributes of the node
        Pooling.update_node_stat(node, attrs)
        return cls.enabled


class AdaptiveAvgPool2dFrontExtractor(FrontExtractorOp):
    op = 'AdaptiveAvgPool2d'
    enabled = True

    @classmethod
    def extract(cls, node):
        Identity.update_node_stat(node)
        return cls.enabled
