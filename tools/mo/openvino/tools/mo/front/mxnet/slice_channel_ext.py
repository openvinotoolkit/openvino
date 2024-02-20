# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.split import AttributedSplit
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs


class SliceChannelFrontExtractor(FrontExtractorOp):
    op = 'SliceChannel'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        axis = attrs.int("axis", 1)
        num_outputs = attrs.int("num_outputs", 0)
        squeeze_axis = attrs.bool('squeeze_axis', False)

        node_attrs = {
            'axis': axis,
            'squeeze_axis': squeeze_axis,
            'num_splits': num_outputs,
        }

        # update the attributes of the node
        AttributedSplit.update_node_stat(node, node_attrs)
        return cls.enabled
