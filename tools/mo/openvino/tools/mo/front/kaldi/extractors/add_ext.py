# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.elementwise import Add
from openvino.tools.mo.front.extractor import FrontExtractorOp


class AddFrontExtractor(FrontExtractorOp):
    op = 'Add'
    enabled = True

    @classmethod
    def extract(cls, node):
        Add.update_node_stat(node, {})
        return cls.enabled
