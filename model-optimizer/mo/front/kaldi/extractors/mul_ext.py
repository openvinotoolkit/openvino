# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.elementwise import Mul
from mo.front.extractor import FrontExtractorOp


class MulFrontExtractor(FrontExtractorOp):
    op = 'Mul'
    enabled = True

    @classmethod
    def extract(cls, node):
        Mul.update_node_stat(node, {})
        return cls.enabled
