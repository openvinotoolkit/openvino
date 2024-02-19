# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp, MXNetCustomFrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs


class CustomFrontExtractorOp(FrontExtractorOp):
    op = 'Custom'
    enabled = True

    def extract(self, node):
        supported = False
        op_attrs = None
        node_attrs = get_mxnet_layer_attrs(node.symbol_dict)
        op_type = node_attrs.str('op_type', None)
        if op_type and op_type in MXNetCustomFrontExtractorOp.registered_ops:
            supported, op_attrs = MXNetCustomFrontExtractorOp.registered_ops[op_type]().extract(node)
        return supported, op_attrs
