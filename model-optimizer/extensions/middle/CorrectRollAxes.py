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
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        roll_ops = graph.get_op_nodes(op='TFRoll')
        for roll in roll_ops:
            correct_roll_axes(roll)
