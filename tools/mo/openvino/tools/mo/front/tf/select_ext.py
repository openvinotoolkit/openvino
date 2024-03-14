# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.select import Select
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.graph.graph import Node


class SelectExtractor(FrontExtractorOp):
    op = 'Select'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Select.update_node_stat(node, {'format': 'tf',})
        return cls.enabled


class SelectV2Extractor(FrontExtractorOp):
    op = 'SelectV2'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Select.update_node_stat(node, {'format': 'tf'})
        return cls.enabled
