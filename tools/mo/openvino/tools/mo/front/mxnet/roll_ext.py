# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.roll import AttributedRoll
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs


class RollExtractor(FrontExtractorOp):
    op = '_np_roll'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        shift = list(attrs.tuple("shift", int, None))
        axis = None
        if attrs.has("axis"):
            axis = list(attrs.tuple("axis", int, None))
        AttributedRoll.update_node_stat(node, {'axes': axis, 'shift': shift})
        return cls.enabled
