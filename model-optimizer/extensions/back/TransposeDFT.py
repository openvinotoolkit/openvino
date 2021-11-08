# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph


class TransposeDFT(BackReplacementPattern):
    """
    In TF models, operation (I)FFTxD has some input shape, [N_0, ..., N_{r - 1}].

    After the transformation SSliceComplexRolledFFTPackBlockReplacement, we have an input shape [N_0, ..., N_{r - 1}, 2]
    for operation DFT or IDFT.

    If the input rank in the TF model was greater than 2, we have [N_0, 2, N_1, ..., N_{r - 1}] as the input shape of
    (I)DFT after the layout conversion, if the option '--disable_nhwc_to_nchw' is not specified.

    But, generally speaking, according to DFT and IDFT specifications, the input shape [N_0, 2, N_1, ..., N_{r - 1}]
    is not correct input shape for DFT and IDFT. Hence, we need to insert Transpose operations before and after (I)DFT
    in such cases.

    This transformation inserts such Transpose nodes, when the source model was the TF model, (I)DFT node has the
    attribute 'need_insert_transposes_for_dft', and this attribute is True.
    """
    enabled = True
    force_shape_inference = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'tf']

    def find_and_replace_pattern(self, graph: Graph):
        import extensions.middle.InsertLayoutPropagationTransposes as InsertTransposes
        for dft in graph.get_op_nodes(need_insert_transposes_for_dft=True):
            InsertTransposes.insert_transpose(graph, dft.in_port(0), before_input=True)
            InsertTransposes.insert_transpose(graph, dft.out_port(0), before_input=False)
