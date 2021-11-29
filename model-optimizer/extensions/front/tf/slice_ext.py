# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.extractor import FrontExtractorOp
from mo.graph.graph import Node
from mo.ops.slice import TFSlice


class SliceExtractor(FrontExtractorOp):
    op = 'Slice'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        TFSlice.update_node_stat(node)
        return cls.enabled
