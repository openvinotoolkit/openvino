# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.ops.const import Const


class ConstantExtractor(FrontExtractorOp):
    op = 'Const'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'data_type': node.value.dtype,
            'value': node.value,
        }
        Const.update_node_stat(node, attrs)
        return cls.enabled
