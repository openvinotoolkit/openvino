# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.ops.sparse_segment_sqrtn import SparseSegmentSqrtN


class SparseSegmentSqrtNFrontExtractor(FrontExtractorOp):
    op = "SparseSegmentSqrtN"
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {}

        SparseSegmentSqrtN.update_node_stat(node, attrs)

        return cls.enabled
