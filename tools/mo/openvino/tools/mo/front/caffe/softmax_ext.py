# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.ops.softmax import Softmax


class SoftmaxFrontExtractor(FrontExtractorOp):
    op = 'Softmax'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        param = proto_layer.softmax_param

        attrs = {
            'axis': param.axis
        }

        # update the attributes of the node
        Softmax.update_node_stat(node, attrs)
        return cls.enabled
