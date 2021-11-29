# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.quantize_linear import QuantizeLinear
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr, get_onnx_opset_version


class QuantizeLinearFrontExtractor(FrontExtractorOp):
    op = 'QuantizeLinear'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {}
        if get_onnx_opset_version(node) >= 13:
            axis = onnx_attr(node, 'axis', 'i', default=None)
            attrs.update(axis=axis)
        QuantizeLinear.update_node_stat(node, attrs)
        return cls.enabled
