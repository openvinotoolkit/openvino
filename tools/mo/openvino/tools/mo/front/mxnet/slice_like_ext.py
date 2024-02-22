# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.slice_like import SliceLike
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs


class SliceLikeFrontExtractor(FrontExtractorOp):
    op = 'slice_like'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        axes = list(attrs.tuple("axes", int, []))
        node_attrs = {
            'axes': axes
        }

        # update the attributes of the node
        SliceLike.update_node_stat(node, node_attrs)
        return cls.enabled
