# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from openvino.tools.mo.ops.softmax import Softmax


class SoftmaxOutputExtractor(FrontExtractorOp):
    op = 'SoftmaxOutput'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)

        axis = 1
        preserve_shape = attrs.str('preserve_shape', 'False')
        multi_output = attrs.str('multi_output', 'False')

        if preserve_shape == 'True':
            axis = -1

        if multi_output == 'True':
            axis = 1

        update_attrs = {
            'axis': axis,
        }

        # update the attributes of the node
        Softmax.update_node_stat(node, update_attrs)
        return cls.enabled
