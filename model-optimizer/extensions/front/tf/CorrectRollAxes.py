# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.tf.graph_utils import correct_roll_axes
from mo.graph.graph import Graph


class CorrectRollAxes(FrontReplacementSubgraph):
    """
    The transformation SSliceComplex removes 2 StridedSlice and Complex operation. If the Roll node is a consumer
    of Complex node in the original TF model, then we have a real input tensor for Roll instead of a complex.
    Negative axes values for the Roll operation should be updated to reflect the fact that the rank of input tensor was
    increased by one (a new trailing dimension of size 2 containing real and imaginary part of complex number is added).
    """
    enabled = True

    def run_after(self):
        from extensions.front.tf.SSliceComplex import SSliceComplex
        return [SSliceComplex]

    def find_and_replace_pattern(self, graph: Graph):
        for roll in graph.get_op_nodes(op='Roll', need_correction=True):
            correct_roll_axes(roll)
            del roll['need_correction']
