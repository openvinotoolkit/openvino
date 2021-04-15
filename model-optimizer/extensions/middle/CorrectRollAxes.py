# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph, Node
from mo.middle.replacement import MiddleReplacementPattern


def correct_roll_axes(roll: Node):
    if not roll.soft_get('need_axes_correction', False):
        return

    axes = roll.in_port(2).data.get_value()
    if axes is None:
        del roll['need_axes_correction']
        return

    corrected_axes = axes.copy()
    for i, axis in enumerate(axes):
        if axis < 0:
            corrected_axes[i] = axis - 1

    roll.in_port(2).data.set_value(int64_array(corrected_axes))
    del roll['need_axes_correction']


class CorrectRollAxes(MiddleReplacementPattern):
    """
    If TFRoll operation was a part of a pattern for the transformation
    StridedSliceComplexRollFFTRollPackBlockReplacement, then the input of the TF Roll was a complex in the TF model,
    and input data shape were [N_0, ..., N_{r - 1}].

    But after the transformation, we have the input data shape [N_0, ..., N_{r - 1}, 2] for TFRoll.

    If 'axes' input contained negative axes, the these indices will be incorrect after the transformation.

    Hence, we should correct these axes. If axis 'a' was negative, then correct new axis is 'a - 1'
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        roll_ops = graph.get_op_nodes(op='TFRoll')
        for roll in roll_ops:
            correct_roll_axes(roll)
