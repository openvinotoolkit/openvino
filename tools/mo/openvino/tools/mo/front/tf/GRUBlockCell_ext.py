# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.ops.GRUBlockCell import GRUBlockCell


class GRUBlockCellExtractor(FrontExtractorOp):
    op = 'GRUBlockCell'
    enabled = True

    @classmethod
    def extract(cls, node):
        GRUBlockCell.update_node_stat(node, {'format': 'tf'})
        return cls.enabled
