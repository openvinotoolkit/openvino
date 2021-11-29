# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.sparse_segment_sum import SparseSegmentSum
from mo.front.extractor import FrontExtractorOp


class SparseSegmentSumFrontExtractor(FrontExtractorOp):
    op = 'SparseSegmentSum'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {}

        SparseSegmentSum.update_node_stat(node, attrs)

        return cls.enabled
