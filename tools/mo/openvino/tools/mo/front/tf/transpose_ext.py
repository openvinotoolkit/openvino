# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.ops.transpose import Transpose


class TransposeFrontExtractorTF(FrontExtractorOp):
    op = "Transpose"
    enabled = True

    @classmethod
    def extract(cls, node):
        Transpose.update_node_stat(node, {"order": None})
        return cls.enabled
