# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.GatherTree import GatherTree
from openvino.tools.mo.front.extractor import FrontExtractorOp


class GatherTreeFrontExtractor(FrontExtractorOp):
    op = 'GatherTree'
    enabled = True

    @classmethod
    def extract(cls, node):
        GatherTree.update_node_stat(node)
        return cls.enabled
