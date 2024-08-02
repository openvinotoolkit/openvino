# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.box_nms import BoxNms
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.graph.graph import Node


class BoxNmsGradExt(FrontExtractorOp):
    op = '_contrib_box_nms'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        BoxNms.update_node_stat(node, {})
        return cls.enabled
