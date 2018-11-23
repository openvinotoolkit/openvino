"""
 Copyright (c) 2018 Intel Corporation

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

from mo.front.common.extractors.utils import layout_attrs
from mo.front.extractor import FrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.ops.pooling import Pooling


class PoolingFrontExtractor(FrontExtractorOp):
    op = 'Pooling'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)

        kernel = attrs.tuple("kernel", int, None)
        stride = attrs.tuple("stride", int, (1, 1))
        padding = attrs.tuple("pad", int, (0, 0))
        method = attrs.str("pool_type", None)
        rt = 'floor'

        data = {
            'window': np.array([1, 1, kernel[0], kernel[1]], dtype=np.int64),
            'stride': np.array([1, 1, stride[0], stride[1]], dtype=np.int64),
            'pad': np.array([[0, 0], [0, 0], [padding[0], padding[0]], [padding[1], padding[1]]], dtype=np.int64),
            'pad_spatial_shape': np.array([[padding[0], padding[0]], [padding[1], padding[1]]], dtype=np.int64),
            'pool_method': method,
            'exclude_pad': 'false',
            'output_spatial_shape': None,
            'rounding_type': rt,
        }

        data.update(layout_attrs())

        pooling_conv = attrs.str("pooling_convention", 'valid')
        if pooling_conv:
            data["pooling_convention"] = pooling_conv
            if pooling_conv == 'full':
                data["rounding_type"] = 'ceil'

        global_pool = attrs.bool("global_pool", False)
        if global_pool:
            data["global_pool"] = global_pool

        # update the attributes of the node
        Pooling.update_node_stat(node, data)
        return __class__.enabled