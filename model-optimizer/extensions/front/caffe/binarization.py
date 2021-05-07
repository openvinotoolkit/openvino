# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.fakequantize import FakeQuantize
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Node, Graph
from mo.ops.const import Const


class BinarizationToQuantize(FrontReplacementOp):
    """
    Replaces Binarization layer with Quantize.
    """
    op = "Binarization"
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        in_node_0 = node.in_node(0)

        broadcast = lambda x: np.array([x], dtype=np.float32)
        threshold = Const(graph, {'name': node.id + "/Input_1", "value": broadcast(0)}).create_node()
        in_1 = threshold
        in_2 = threshold
        in_3 = Const(graph, {'name': node.id + "/Input_3", "value": broadcast(-1)}).create_node()
        in_4 = Const(graph, {'name': node.id + "/Input_4", "value": broadcast(+1)}).create_node()
        quant = FakeQuantize(graph, {'name': node.id + "/FakeQuantize_", "levels": 2}).create_node(
            inputs=[in_node_0, in_1, in_2, in_3, in_4])

        return [quant.id]
