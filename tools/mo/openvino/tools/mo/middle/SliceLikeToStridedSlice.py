# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from typing import Dict

from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, Node, rename_nodes
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.shape import Shape
from openvino.tools.mo.ops.strided_slice import StridedSlice
from openvino.tools.mo.utils.shape import get_shape_values_by_range_idxs, new_shape_node_from_shape_nodes, \
    get_shape_and_rank_nodes_by_port


class SliceLikeToStridedSlice(MiddleReplacementPattern):
    """
    Replace mxnet slice_like operation with StridedSlice in reshapable way.
    The begin parameter for StridedSlice is always a zero vector.
    The end parameter depends on the slice_like inputs and axes.

    1. If slice_like inputs has the same ranks, we can use second input shape (shape_like) as the end parameter for
       StridedSlice. Axes parameter will form end_mask, that allows to use slice only on the desired axes.
       Example:
       input_shape = [1, 64, 128, 256], shape_like = [1, 2, 3, 4], axes = [2, 3].
       In that case end = shape_like = [1, 2, 3, 4], but end_mask = [0, 0, 1, 1], so output_shape = [1, 64, 3, 4]

    2. Axes parameter has the last dimension of the first input shape (in that case shape_like >= input_shape).
       Here we can use only a part of shape_like as the end parameter.
       Example:
           input_shape = [1, 64, 128, 256], shape_like = [1, 2, 3, 4, 5], axes = [2, 3].
           end = shape_like[:4] = [1, 2, 3, 4], end_mask = [0, 0, 1, 1], output_shape = [1, 64, 3, 4]

    3. Usual case, where we form end parameter by concatenate parts of shape_like and input_shape.
       Examples:
           input_shape = [1, 64, 128, 256, 512], shape_like = [1, 2, 3, 4], axes = [2, 3].
           end = shape_like[:4] + input_shape[4:] = [1, 2, 3, 4, 512],
           end_mask = [0, 0, 1, 1, 0], output_shape = [1, 64, 3, 4, 512]

           input_shape = [1, 64, 128, 256], shape_like = [1, 2, 3, 4, 5], axes = [0, 2].
           end = shape_like[:3] + input_shape[3:] = [1, 2, 3, 256],
           end_mask = [1, 0, 1, 0], output_shape = [1, 64, 3, 256]
    """

    enabled = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'mxnet']

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('op', dict(kind='op', op='slice_like'))
            ],
            edges=[]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: Dict[str, Node]):
        node = match['op']
        name = node.soft_get('name', node.id)
        input_shape = node.in_port(0).data.get_shape()
        second_input_shape = node.in_port(1).data.get_shape()

        begin_mask = np.zeros(len(input_shape), dtype=np.int64)
        end_mask = np.zeros(len(input_shape), dtype=np.int64)

        for i in node.axes:
            end_mask[i] = np.int64(1)

        new_axis_mask = np.zeros(len(input_shape), dtype=np.int64)
        shrink_axis_mask = np.zeros(len(input_shape), dtype=np.int64)
        ellipsis_mask = np.zeros(len(input_shape), dtype=np.int64)

        ss = create_op_with_const_inputs(graph, StridedSlice,
                                         port_value_dict={1: np.zeros(len(input_shape), dtype=np.int64)},
                                         op_attrs={'name': 'StridedSlice', 'begin_mask': begin_mask,
                                                   'end_mask': end_mask, 'new_axis_mask': new_axis_mask,
                                                   'shrink_axis_mask': shrink_axis_mask,
                                                   'ellipsis_mask': ellipsis_mask})
        if input_shape.size == second_input_shape.size:
            end = Shape(graph, dict(name=name + '/End')).create_node()
            end.in_port(0).connect(node.in_port(1).get_source())
            ss.in_port(2).connect(end.out_port(0))
        else:
            shape_like, rank_like = get_shape_and_rank_nodes_by_port(node.in_port(1).get_source())
            end_first_part = get_shape_values_by_range_idxs(shape_like, rank_like, 0, node.axes[-1], include_end=True)
            if input_shape.size - 1 == node.axes[-1]:
                ss.in_port(2).connect(end_first_part.out_port(0))
            else:
                shape, rank = get_shape_and_rank_nodes_by_port(node.in_port(0).get_source())
                end_second_part = get_shape_values_by_range_idxs(shape, rank, node.axes[-1], -1, include_begin=False,
                                                                 include_end=True)
                end = new_shape_node_from_shape_nodes([end_first_part, end_second_part])
                ss.in_port(2).connect(end.out_port(0))

        node.in_port(0).get_connection().set_destination(ss.in_port(0))
        node.in_port(1).disconnect()
        node.out_port(0).get_connection().set_source(ss.out_port(0))

        rename_nodes([(node, name + '/ShouldBeDeleted'), (ss, name)])
