# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.elemental import copy_shape_infer
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.graph.graph import Node


class NextIterationExtractor(FrontExtractorOp):
    op = "NextIteration"
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        node['is_cyclic'] = True
        node['infer'] = copy_shape_infer
        return cls.enabled
