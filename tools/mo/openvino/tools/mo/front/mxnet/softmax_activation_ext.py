# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from openvino.tools.mo.ops.softmax import Softmax


class SoftmaxActivationExtractor(FrontExtractorOp):
    op = 'SoftmaxActivation'
    enabled = True

    @classmethod
    def extract(cls, node):
        attr = get_mxnet_layer_attrs(node.symbol_dict)
        mode = attr.str("mode", "instance")

        if mode == "channel":
            axis = 1
        else:
            axis = -1

        update_attrs = {
            'axis': axis,
        }

        # update the attributes of the node
        Softmax.update_node_stat(node, update_attrs)
        return cls.enabled
