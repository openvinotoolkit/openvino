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


def global_average_pool_ext(node):

    pooling_convention = 'full'
    rt = 'ceil'

    attrs = {
        'type': 'Pooling',
        'window': np.array([1, 1, 0, 0], dtype=np.int64),
        'stride': np.array([1, 1, 1, 1], dtype=np.int64),
        'pad': np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int64),
        'pad_spatial_shape': np.array([[0, 0], [0, 0]], dtype=np.int64),
        'pool_method': 'avg',
        'exclude_pad': 'false',
        'infer': pool_explicit_padding_infer,
        'global_pool': True,
        'output_spatial_shape': None,
        'rounding_type': rt
    }

    attrs.update(layout_attrs())
    attrs['pooling_convention'] = pooling_convention
    return attrs

