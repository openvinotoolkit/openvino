# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.MatMul import MatMul


class BatchDotExt(FrontExtractorOp):
    """
    MXNet operation which compute dot product of x and y, where x and y are data in batch: [B_0, B_1, ..., B_N, :, :].
    Two right-most axes in tensor are interpreted as matrix rows and columns dimensions.

    Attributes:
        transpose_a - if true then transpose the first input before dot
        transpose_b - if true the transpose the second input before dot
    """
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
