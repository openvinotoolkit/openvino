# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.roll import Roll
from openvino.tools.mo.front.extractor import FrontExtractorOp


class RollExtractor(FrontExtractorOp):
    op = 'Roll'
    enabled = True

    @classmethod
    def extract(cls, node):
        Roll.update_node_stat(node, {})
        return cls.enabled
