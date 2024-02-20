# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.activation_ops import ReLU
from openvino.tools.mo.front.extractor import FrontExtractorOp


class ReLUFrontExtractor(FrontExtractorOp):
    op = 'relu'
    enabled = True

    @classmethod
    def extract(cls, node):
        ReLU.update_node_stat(node)
        return ReLUFrontExtractor.enabled
