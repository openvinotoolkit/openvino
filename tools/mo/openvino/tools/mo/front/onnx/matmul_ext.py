# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.MatMul import MatMul
from openvino.tools.mo.front.extractor import FrontExtractorOp


class MatMulFrontExtractor(FrontExtractorOp):
    op = 'MatMul'
    enabled = True

    @classmethod
    def extract(cls, node):
        MatMul.update_node_stat(node)
        return cls.enabled
