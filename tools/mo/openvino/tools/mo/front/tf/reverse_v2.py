# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.Reverse import Reverse
from openvino.tools.mo.front.extractor import FrontExtractorOp


class ReverseV2FrontExtractor(FrontExtractorOp):
    op = 'ReverseV2'
    enabled = True

    @classmethod
    def extract(cls, node):
        Reverse.update_node_stat(node)
        return cls.enabled
