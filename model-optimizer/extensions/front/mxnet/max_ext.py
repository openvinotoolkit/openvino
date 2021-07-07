# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.ReduceOps import ReduceMax
from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import FrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs


class MaxFrontExtractor(FrontExtractorOp):
    op = 'max'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        ReduceMax.update_node_stat(node, {'axis': int64_array([attrs.int('axis', 0)]), 'keep_dims': False})
        return cls.enabled
