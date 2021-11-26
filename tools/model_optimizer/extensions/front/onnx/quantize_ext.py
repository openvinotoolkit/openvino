# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.fakequantize import FakeQuantize
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class FakeQuantizeFrontExtractor(FrontExtractorOp):
    op = 'FakeQuantize'
    enabled = True

    @classmethod
    def extract(cls, node):
        levels = onnx_attr(node, 'levels', 'i')
        FakeQuantize.update_node_stat(node, {'levels': levels})
        return FakeQuantizeFrontExtractor.enabled
