# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.elemental import single_output_infer
from openvino.tools.mo.front.extractor import FrontExtractorOp


class LoopCondFrontExtractor(FrontExtractorOp):
    op = 'LoopCond'
    enabled = True

    @classmethod
    def extract(cls, node):
        node['infer'] = lambda node: single_output_infer(
            node,
            lambda node: node.in_node(0).shape,
            lambda node: node.in_node(0).value
        )
        return cls.enabled
