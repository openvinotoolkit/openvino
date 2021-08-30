# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.MatMul import MatMul
from mo.front.extractor import FrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.graph.graph import Node


class BatchDotExt(FrontExtractorOp):
    op = 'batch_dot'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        transpose_a = attrs.bool('transpose_a', False)
        transpose_b = attrs.bool('transpose_b', False)

        MatMul.update_node_stat(node, {
            'transpose_a': transpose_a,
            'transpose_b': transpose_b
        })
        return cls.enabled
