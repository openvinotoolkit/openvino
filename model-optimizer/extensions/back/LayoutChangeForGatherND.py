# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph


class LayoutChangeForGatherND(BackReplacementPattern):
    """
    Return original layout for inputs and output of GatherND operation
    since the operation is designed for NHWC layout.
    """
    enabled = True
    force_shape_inference = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'tf']

    def find_and_replace_pattern(self, graph: Graph):
        import extensions.middle.InsertLayoutPropagationTransposes as InsertTransposes
        for gathernd in graph.get_op_nodes(type='GatherND'):
            InsertTransposes.insert_transpose(graph, gathernd.in_port(0), before_input=True)
            InsertTransposes.insert_transpose(graph, gathernd.in_port(1), before_input=True)
            InsertTransposes.insert_transpose(graph, gathernd.out_port(0), before_input=False)
