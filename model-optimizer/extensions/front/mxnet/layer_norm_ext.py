# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.layer_norm import LayerNorm
from mo.front.extractor import FrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.graph.graph import Node


class LayerNormFrontExtractor(FrontExtractorOp):
    op = 'LayerNorm'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        attr = get_mxnet_layer_attrs(node.symbol_dict)

        node_attrs = {
            'epsilon': attr.float('eps', 1e-5),
            'axis': attr.int('axis', -1),
        }
        LayerNorm.update_node_stat(node, node_attrs)
        return cls.enabled
