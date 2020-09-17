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
from mo.utils.error import Error


class MaxPool2dFrontExtractor(FrontExtractorOp):
    op = 'MaxPool2d'
    enabled = True

    @classmethod
    def extract(cls, node):
        # # Extract pads attribute
        # # In case if pads is not specified it will be set in default (1) in infer function
        # pads = onnx_attr(node, 'pads', 'ints', default=None, dst_type=lambda x: np.array(x, dtype=np.int64))
        # assert pads is None or len(pads) % 2 == 0
        # pads = np.array(node.module.padding, dtype=np.int64).reshape(2, -1)

        print(node.module.kernel_size)

        pads = np.array([node.module.padding, node.module.padding], dtype=np.int64).reshape(1, 2)
        pads = np.repeat(pads, 2, axis=0)
        final_pads = np.array([[0, 0], [0, 0], *pads], dtype=np.int64)

        # # Extract dilations attribute
        # # In case if dilations is not specified it will be set in default (1) in infer function
        # dilations = onnx_attr(node, 'dilations', 'ints', default=None, dst_type=lambda x: np.array(x, dtype=np.int64))
        # final_dilations = np.array([1, 1, *dilations], dtype=np.int64) if dilations is not None else None
        #
        # # Extract dilations attribute
        # # In case if dilations is not specified it will be set in default (1) in infer function
        strides = [node.module.stride, node.module.stride]
        final_strides = np.array([1, 1, *strides], dtype=np.int64)
        print(final_strides)

        kernel_shape = [node.module.kernel_size, node.module.kernel_size]
        final_kernel_shape = np.array([1, 1, *kernel_shape], dtype=np.int64)
        print(final_kernel_shape)

        attrs = {
            'op': node.op,
            # 'auto_pad': auto_pad,
            'window': final_kernel_shape,
            'stride': final_strides,
            'pad': final_pads,
            # 'pad_spatial_shape': np.array(pads, dtype=np.int64) if pads is not None else None,
            'pool_method': 'max',
            # 'exclude_pad': 'true' if exclude_pad else 'false',
            # 'global_pool': global_pooling,
            # 'output_spatial_shape': None,
            # 'rounding_type': rt,
            #
            # 'spatial_dims': None,
            'channel_dims': np.array([1], dtype=np.int64),
            'batch_dims': np.array([0], dtype=np.int64),
            # 'layout': 'NCHW',
            #
            # 'pooling_convention': pooling_convention
        }

        # update the attributes of the node
        Pooling.update_node_stat(node, attrs)
        return cls.enabled
