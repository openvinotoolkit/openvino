# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.split import VariadicSplit, Split
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph


class SliceToVariadicSplit(FrontReplacementOp):
    op = "CaffeSlice"
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['op']
        name = node.soft_get('name', node.id)

        assert node.has_valid('axis'), 'Slice operation `{}` has no `axis` parameter'.format(name)
        axis = int64_array(node.axis)
        if axis.size != 1:
            return

        assert node.has_valid('slice_point'), 'Slice operation `{}` has no `slice_point` parameter'.format(name)
        slice_point = node.slice_point

        if slice_point.size == 0:
            num_splits = len(node.out_ports())
            split_node = create_op_with_const_inputs(graph, op=Split,
                                                     port_value_dict={1: axis},
                                                     op_attrs={'name': name, 'num_splits': num_splits})
        else:
            size_splits = []
            curr_pos = 0
            for point in slice_point:
                assert point > curr_pos
                size_splits.append(point - curr_pos)
                curr_pos = point
            size_splits.append(-1)

            split_node = create_op_with_const_inputs(graph, op=VariadicSplit,
                                                     port_value_dict={1: axis, 2: int64_array(size_splits)},
                                                     op_attrs={'name': name, 'out_ports_count': len(slice_point) + 1})

        node.in_port(0).get_connection().set_destination(split_node.in_port(0))
        for i, port in node.out_ports().items():
            node.out_port(i).get_connection().set_source(split_node.out_port(i))
