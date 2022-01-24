# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#! [complex_abs:transformation]
import numpy as np

from openvino.tools.mo.ops.elementwise import Pow
from openvino.tools.mo.ops.ReduceOps import ReduceSum
from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.ops.const import Const


class ComplexAbs(FrontReplacementOp):
    op = "ComplexAbs"
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        pow_2 = Const(graph, {'value': np.float32(2.0)}).create_node()
        reduce_axis = Const(graph, {'value': np.int32(-1)}).create_node()
        pow_0_5 = Const(graph, {'value': np.float32(0.5)}).create_node()

        sq = Pow(graph, dict(name=node.in_node(0).name + '/sq', power=2.0)).create_node([node.in_node(0), pow_2])
        sum = ReduceSum(graph, dict(name=sq.name + '/sum')).create_node([sq, reduce_axis])
        sqrt = Pow(graph, dict(name=sum.name + '/sqrt', power=0.5)).create_node([sum, pow_0_5])
        return [sqrt.id]
#! [complex_abs:transformation]
