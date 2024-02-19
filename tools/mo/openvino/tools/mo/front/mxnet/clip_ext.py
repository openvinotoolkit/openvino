# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.clamp import AttributedClamp


class ClipExt(FrontExtractorOp):
    op = 'clip'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)

        AttributedClamp.update_node_stat(node, {'min': attrs.float('a_min', None), 'max': attrs.float('a_max', None)})
        return cls.enabled
