# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.elementwise import Mul
from openvino.tools.mo.front.extractor import FrontExtractorOp


class MulFrontExtractor(FrontExtractorOp):
    op = 'Mul'
    enabled = True

    @classmethod
    def extract(cls, node):
        Mul.update_node_stat(node, {})
        return cls.enabled
