# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.gather import AttributedGather
from mo.front.extractor import FrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.graph.graph import Node


class TakeExtractor(FrontExtractorOp):
    op = 'take'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        AttributedGather.update_node_stat(node, {
            'axis': attrs.int('axis', 0),
        })
        return cls.enabled
