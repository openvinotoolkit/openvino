# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.range import Range
from mo.front.extractor import FrontExtractorOp
from mo.graph.graph import Node


class RangeFrontExtractor(FrontExtractorOp):
    op = 'Range'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        # output_type attribute will be deduced during shape infer
        Range.update_node_stat(node, {})
        return cls.enabled

