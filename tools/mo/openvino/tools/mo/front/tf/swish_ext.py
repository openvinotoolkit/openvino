# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.activation_ops import Swish
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.graph.graph import Node


class SwishExtractor(FrontExtractorOp):
    op = 'swish_f32'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Swish.update_node_stat(node, {})
        return cls.enabled

