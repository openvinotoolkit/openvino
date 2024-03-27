# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.extractors.utils import layout_attrs
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.kaldi.loader.utils import read_token_value, collect_until_token, \
    read_binary_integer32_token, find_next_tag, read_placeholder
from openvino.tools.mo.ops.pooling import Pooling
from openvino.tools.mo.utils.error import Error


class MaxPoolingComponentFrontExtractor(FrontExtractorOp):
    op = 'maxpoolingcomponent'
    enabled = True

    @classmethod
    def extract(cls, node):
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
            'window': int64_array([1, 1, 1, kernel]),
            'stride': int64_array([1, 1, 1, stride]),
            'pool_stride': pool_stride,
            'pool_step': pool_step,
            'pad': int64_array([[0, 0], [0, 0], [0, 0], [0, 0]]),
            'pad_spatial_shape': int64_array([[0, 0], [0, 0]]),
            'pool_method': 'max',
        }
        mapping_rule.update(layout_attrs())
        Pooling.update_node_stat(node, mapping_rule)
        return cls.enabled
