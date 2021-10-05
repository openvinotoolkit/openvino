# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.extractor import FrontExtractorOp
from mo.ops.const import Const


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
