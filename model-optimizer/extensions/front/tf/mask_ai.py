# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementPattern
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.middle.pattern_match import apply_pattern
from mo.ops.const import Const
from mo.ops.reshape import Reshape

MASK_EIP_PATTERN = {
    'nodes': [
        ('eip', dict(op='ExtractImagePatches')),
        ('reshape', dict(op='Reshape')),
        ('ss', dict(op='StridedSlice')),
        ('mean', dict(op='ReduceMean')),
        ('reshape_const', dict(op='Const')),
        ('mean_const', dict(op='Const')),
    ],
    'edges': [
        ('eip', 'reshape', {'out': 0, 'in': 0}),
        ('reshape_const', 'reshape', {'out': 0, 'in': 1}),
        ('reshape', 'ss', {'out': 0, 'in': 0}),
        ('ss', 'mean', {'out': 0, 'in': 0}),
        ('mean_const', 'mean', {'out': 0, 'in': 1}),
    ]
}

class MaskTransformation(FrontReplacementPattern):
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        apply_pattern(graph, **MASK_EIP_PATTERN, action=self.optimize_eip)

    @staticmethod
    def optimize_eip(graph: Graph, match: dict):
        reshape_const = match['reshape_const']
        mean_const = match['mean_const']
        if not np.array_equal(reshape_const.value, [1, -1, 3, 3, 1]):
            return
        if not np.array_equal(mean_const.value, [2, 3]):
            return

        mean = match['mean']
        eip = match['eip']

        mean.in_port(0).disconnect()
        eip.out_port(0).get_connection().set_destination(mean.in_port(0))

        mean.in_port(1).disconnect()
        mean.in_port(1).connect(Const(graph, {'value': int64_array([-1])}).create_node().out_port(0))
        mean.out_port(0).get_connection().insert_node(
            create_op_node_with_second_input(graph, Reshape, int64_array([1, -1, 1, 1])))

        graph.remove_nodes_from([match['ss'].id])
