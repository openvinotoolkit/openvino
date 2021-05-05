# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.back.ReshapeMutation import ReshapeMutation
from mo.back.replacement import BackReplacementPattern
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.ops.reshape import Reshape


# Temporary nGraph workaround. TODO: REMOVE
class ScalarNormalize(BackReplacementPattern):
    enabled = False
    force_clean_up = True

    def run_before(self):
        return [ReshapeMutation]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('op', dict(kind='op', type='Const'))],
            edges=[]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['op']
        if node.value.ndim == 0:
            reshape = create_op_node_with_second_input(graph, Reshape, int64_array([1]),
                                                       {'name': node.id + '/Dims'})
            node.out_port(0).get_connection().set_source(reshape.out_port(0))
            node.out_port(0).connect(reshape.in_port(0))
            reshape.infer(reshape)
