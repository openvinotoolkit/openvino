# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.extractor import FrontExtractorOp
from mo.ops.fill import Fill


class FillExtractor(FrontExtractorOp):
    op = 'Fill'
    enabled = True

    @classmethod
    def extract(cls, node):
        Fill.update_node_stat(node, {})
        return cls.enabled
