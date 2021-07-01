# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.cumsum import CumSum
from mo.front.extractor import FrontExtractorOp


class CumSumExtractor(FrontExtractorOp):
    op = 'Cumsum'
    enabled = True

    @classmethod
    def extract(cls, node):
        exclusive = node.pb.attr['exclusive'].b
        reverse = node.pb.attr['reverse'].b
        CumSum.update_node_stat(node, {'exclusive': exclusive, 'reverse': reverse})
        return cls.enabled
