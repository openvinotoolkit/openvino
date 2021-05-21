# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.sparse_fill_empty_rows import SparseFillEmptyRows
from mo.front.extractor import FrontExtractorOp


class SparseFillEmptyRowsFrontExtractor(FrontExtractorOp):
    op = 'SparseFillEmptyRows'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {}

        SparseFillEmptyRows.update_node_stat(node, attrs)

        return cls.enabled
