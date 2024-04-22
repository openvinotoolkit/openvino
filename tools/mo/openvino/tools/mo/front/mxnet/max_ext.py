# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.ReduceOps import ReduceMax
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs


class MaxFrontExtractor(FrontExtractorOp):
    op = 'max'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        ReduceMax.update_node_stat(node, {'axis': int64_array([attrs.int('axis', 0)]), 'keep_dims': False})
        return cls.enabled
