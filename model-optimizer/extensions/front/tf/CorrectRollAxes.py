# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.tf.graph_utils import correct_roll_axes
from mo.graph.graph import Graph


class CorrectRollAxes(FrontReplacementSubgraph):
    """
    After the transformation SSliceComplex, if Roll node was after Complex node in the source TF model,
    we have a real input tensor for Roll instead of a complex code. Hence, if axes were negative, then
    these axes were incorrect. As a sequence, axes must be corrected.
    """
    enabled = True

    def run_after(self):
        from extensions.front.tf.SSliceComplex import SSliceComplex
        return [SSliceComplex]

    def find_and_replace_pattern(self, graph: Graph):
        for roll in graph.get_op_nodes(op='Roll', need_correction=True):
            correct_roll_axes(roll)
            del roll['need_correction']
