# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph


class CorrectRollAxes(FrontReplacementOp):
    op = 'Roll'
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        roll_node = match['op']
        if not roll_node.soft_get('need_axes_correction', False):
            return

        axes = roll_node.in_port(0).data.get_value()
        if axes is None:
            del roll_node['need_axes_correction']
            return

        corrected_axes = axes.copy()
        for i, axis in enumerate(axes):
            if axis < 0:
                corrected_axes[i] = axis - 1

        roll_node.in_port(1).data.set_value(int64_array(corrected_axes))
        del roll_node['need_axes_correction']
