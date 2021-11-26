# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.BN import BN
from mo.front.extractor import FrontExtractorOp


class BNExtractor(FrontExtractorOp):
    op = 'BN'
    enabled = True

    @classmethod
    def extract(cls, node):
        BN.update_node_stat(node, {})
        return cls.enabled
