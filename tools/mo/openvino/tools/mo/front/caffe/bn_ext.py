# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.BN import BN
from openvino.tools.mo.front.extractor import FrontExtractorOp


class BNExtractor(FrontExtractorOp):
    op = 'BN'
    enabled = True

    @classmethod
    def extract(cls, node):
        BN.update_node_stat(node, {})
        return cls.enabled
