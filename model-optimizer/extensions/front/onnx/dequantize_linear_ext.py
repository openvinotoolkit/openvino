# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from extensions.ops.dequantize_linear import DequantizeLinear
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr, get_onnx_opset_version


class DequantizeLinearFrontExtractor(FrontExtractorOp):
    op = 'DequantizeLinear'
    enabled = True

    @classmethod
    def extract(cls, node):
        if get_onnx_opset_version(node) >= 13:
            log.warning('Ignoring "axis" attribute for DequantizeLinear-{} node, inference might be incorrect.'.format(
                get_onnx_opset_version(node)))
        DequantizeLinear.update_node_stat(node)
        return cls.enabled
