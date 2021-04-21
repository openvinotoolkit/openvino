# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph


class TransposeDFT(BackReplacementPattern):
    """
    Return original layout for inputs and output of (I)DFT operation when this operation came from TF.
    """
    enabled = True
    force_shape_inference = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'tf']

    def find_and_replace_pattern(self, graph: Graph):
        import extensions.middle.InsertLayoutPropagationTransposes as InsertTransposes
        for dft in graph.get_op_nodes(need_insert_transposes_for_dft=True):
            InsertTransposes.insert_transpose(graph, dft.in_port(0), before_input=True)
            InsertTransposes.insert_transpose(graph, dft.out_port(0), before_input=False)
