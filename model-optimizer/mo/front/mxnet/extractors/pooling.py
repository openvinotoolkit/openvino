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
from mo.front.common.partial_infer.pooling import pool_explicit_padding_infer


def pooling_ext(attrs):
    kernel = attrs.tuple("kernel", int, None)
    stride = attrs.tuple("stride", int, (1, 1))
    padding = attrs.tuple("pad", int, (0, 0))
    method = attrs.str("pool_type", None)

    data = {
        'window': np.array([1, 1, kernel[1], kernel[0]], dtype=np.int64),
        'stride': np.array([1, 1, stride[1], stride[0]], dtype=np.int64),
        'pad': np.array([[0, 0], [0, 0], [padding[1], padding[1]], [padding[0], padding[0]]], dtype=np.int64),
        'pad_spatial_shape': np.array([[padding[1], padding[1]], [padding[0], padding[0]]], dtype=np.int64),
        'pool_method': method,
        'exclude_pad': 'false',
        'infer': pool_explicit_padding_infer,
        'output_spatial_shape': None,
        'rounding_type': 'floor'
    }

    data.update(layout_attrs())

    pooling_conv = attrs.str("pooling_convention", 'valid')
    if pooling_conv:
        data["pooling_convention"] = pooling_conv
        data["rounding_type"] = 'ceil'

    global_pool = attrs.bool("global_pool", False)
    if global_pool:
        data["global_pool"] = global_pool

    return data
