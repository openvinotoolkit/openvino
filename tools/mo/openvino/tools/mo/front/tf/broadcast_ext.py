# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.broadcast import Broadcast


class BroadcastExtractor(FrontExtractorOp):
    op = 'BroadcastTo'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Broadcast.update_node_stat(node, attrs={'mode': 'numpy'})
        return cls.enabled
