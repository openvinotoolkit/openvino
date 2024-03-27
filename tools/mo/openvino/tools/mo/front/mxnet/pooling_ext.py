# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from openvino.tools.mo.ops.pooling import Pooling


class PoolingFrontExtractor(FrontExtractorOp):
    op = 'Pooling'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)

        kernel = attrs.tuple("kernel", int, None)
        stride = attrs.tuple("stride", int, tuple(np.ones(len(kernel), dtype=np.int64)))
        padding = attrs.tuple("pad", int, tuple(np.zeros(len(kernel), dtype=np.int64)))
        method = attrs.str("pool_type", None)
        rt = 'floor'

        data = {
            'window': int64_array([1, 1, *[k for k in kernel]]),
            'stride': int64_array([1, 1, *[s for s in stride]]),
            'pad': int64_array([[0, 0], [0, 0], *[[pad, pad] for pad in padding]]),
            'pad_spatial_shape': int64_array([[pad, pad] for pad in padding]),
            'pool_method': method,
            'exclude_pad': False,
            'output_spatial_shape': None,
            'spatial_dims': None,
            'channel_dims': int64_array([1]),
            'batch_dims': int64_array([0]),
            'layout': 'NCHW',
            'rounding_type': rt,
        }

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
        return cls.enabled
