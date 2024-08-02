# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.sparse_segment_sum import SparseSegmentSum
from openvino.tools.mo.front.extractor import FrontExtractorOp


class SparseSegmentSumFrontExtractor(FrontExtractorOp):
    op = 'SparseSegmentSum'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {}

        SparseSegmentSum.update_node_stat(node, attrs)

        return cls.enabled
