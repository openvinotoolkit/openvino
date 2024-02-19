# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.ops.broadcast import Broadcast


class ExpandExtractor(FrontExtractorOp):
    op = 'Expand'
    enabled = True

    @classmethod
    def extract(cls, node):
        Broadcast.update_node_stat(node, {'mode': 'bidirectional'})
        return cls.enabled
