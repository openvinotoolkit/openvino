"""
 Copyright (c) 2018-2019 Intel Corporation

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
from mo.front.kaldi.loader.utils import read_token_value, collect_until_whitespace, collect_until_token, \
    read_binary_integer32_token, find_next_tag, read_placeholder
from mo.ops.pooling import Pooling
from mo.utils.error import Error


class MaxPoolingComponentFrontExtractor(FrontExtractorOp):
    op = 'maxpoolingcomponent'
    enabled = True

    @staticmethod
    def extract(node):
        pb = node.parameters
        collect_until_token(pb, b'<PoolSize>')
        kernel = read_binary_integer32_token(pb)
        tag = find_next_tag(pb)
        if tag == '<PoolStep>':
            read_placeholder(pb, 1)
            stride = read_binary_integer32_token(pb)
            pool_step = stride
            pool_stride = read_token_value(pb, b'<PoolStride>')
        elif tag == '<PoolStride>':
            stride = 1
            pool_step = None
            read_placeholder(pb, 1)
            pool_stride = read_binary_integer32_token(pb)
        else:
            raise Error('Can not extract parameters for {}'.format(node))

        mapping_rule = {
            'window': np.array([1, 1, 1, kernel], dtype=np.int64),
            'stride': np.array([1, 1, stride, stride], dtype=np.int64),
            'pool_stride': pool_stride,
            'pool_step': pool_step,
            'pad': np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int64),
            'pad_spatial_shape': np.array([[0, 0], [0, 0]], dtype=np.int64),
            'pool_method': 'max',
        }
        mapping_rule.update(layout_attrs())
        Pooling.update_node_stat(node, mapping_rule)
        return __class__.enabled
