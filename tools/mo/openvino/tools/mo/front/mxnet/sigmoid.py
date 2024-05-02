# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.ops.activation_ops import Sigmoid


class SigmoidFrontExtractor(FrontExtractorOp):
    op = "sigmoid"
    enabled = True

    @classmethod
    def extract(cls, node):
        Sigmoid.update_node_stat(node)
        return cls.enabled
