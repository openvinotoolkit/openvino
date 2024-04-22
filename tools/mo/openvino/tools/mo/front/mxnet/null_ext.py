# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.parameter import Parameter
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.ops.const import Const


class NullFrontExtractor(FrontExtractorOp):
    op = 'null'
    enabled = True

    @classmethod
    def extract(cls, node):
        if 'value' in node.symbol_dict:
            Const.update_node_stat(node, {'value': node.symbol_dict['value']})
        else:
            Parameter.update_node_stat(node, {})
        return cls.enabled
