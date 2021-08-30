# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.gather import Gather
from extensions.ops.range import Range
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementOp
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph
from mo.ops.shape import Shape


class ArangeLikeReplacer(FrontReplacementOp):
    op = 'arange_like'
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['op']
        name = node.soft_get('name')
        axis = node.soft_get('axis')
        if axis:
            shape = Shape(graph, {'name': name + '/shape'}).create_node()
            gather_node = create_op_with_const_inputs(graph, Gather, {1: int64_array([axis]),
                                                                      2: int64_array(0)},
                                                      {'name': name + '/Gather'})
            range_node = create_op_with_const_inputs(graph, Range, {0: int64_array([0]),
                                                                    2: int64_array([1])},
                                                     {'name': name + '/Range', 'output_type': np.int32})
            shape.in_port(0).connect(node.in_port(0).get_source())
            shape.out_port(0).connect(gather_node.in_port(0))
            gather_node.out_port(0).connect(range_node.in_port(1))
            node.out_port(0).get_connection().set_source(range_node.out_port(0))
            node.in_port(0).disconnect()
            print("OK")
