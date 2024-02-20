# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from openvino.tools.mo.ops.concat import Concat


class RNNParamConcatFrontExtractor(FrontExtractorOp):
    op = '_rnn_param_concat'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        data = {
            'axis': attrs.int("dim", 1),
        }

        # update the attributes of the node
        Concat.update_node_stat(node, data)
        return cls.enabled
