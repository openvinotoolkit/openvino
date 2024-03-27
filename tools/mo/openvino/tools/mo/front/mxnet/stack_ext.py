# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.pack import PackOp
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs


class StackFrontExtractor(FrontExtractorOp):
    op = 'stack'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)

        update_attrs = {
            'axis': attrs.int('axis', 0)
        }

        # update the attributes of the node
        PackOp.update_node_stat(node, update_attrs)

        return cls.enabled
