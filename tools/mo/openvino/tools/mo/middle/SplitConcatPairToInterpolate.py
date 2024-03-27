# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
from typing import Optional

import numpy as np

from openvino.tools.mo.ops.Cast import Cast
from openvino.tools.mo.ops.activation_ops import Floor
from openvino.tools.mo.ops.elementwise import Mul
from openvino.tools.mo.ops.interpolate import Interpolate
from openvino.tools.mo.front.common.partial_infer.utils import int64_array, float32_array
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.shape import Shape
from openvino.tools.mo.ops.strided_slice import StridedSlice


def get_concat_after_split(split: Node) -> Optional[Node]:
    """
    This function gets consumers of the 'split' node, checks that the following conditions are fulfilled:
    1) 'split' node has only one consumer;
    2) for any output port of 'split', number of corresponding input ports of the consumer is the same;
    3) for any output port 'i' of the 'split', corresponding input ports of the consumer are
       [i * m, ..., i * m + (m - 1)], where 'm' is the same for all 'i';
    4) the consumer operation is 'Concat';
    5) 'split' is a unique producer for this 'Concat';
    and, if all these conditions are fulfilled, returns the above mentioned 'Concat' node. Otherwise, if some of these
    conditions is false, this functions returns None.
    :param split: Split node
    :return: Concat node, if all conditions are fulfilled, or None otherwise
    """
    # If number of output nodes of 'split' is not equal to 1, then the transformation is not applicable.
    split_outputs = [d.node for _, p in split.out_ports().items() for d in p.get_connection().get_destinations()]
    names_of_split_outputs = set([n.name for n in split_outputs])
    if len(names_of_split_outputs) != 1:
        return

    groups_of_inputs = [[d.idx for d in p.get_connection().get_destinations()] for _, p in split.out_ports().items()]
    sizes_of_groups = set([len(g) for g in groups_of_inputs])
    # If numbers of consumer ports are various for various output ports of 'split', then the transformation
    # is not applicable.
    if len(sizes_of_groups) != 1:
        return
    # The transformation is applicable iff output port 0 of 'split' goes to ports [0, ..., m-1] of next node,
    # output port 1 of 'split' goes to ports [m, ..., m + (m-1)] of next node, ..., output port i of 'split'
    # goes to ports [i * m, ..., i * m + (m - 1)], and so on.
    flatten_groups = [i for g in groups_of_inputs for i in g]
    if flatten_groups != list(range(0, len(flatten_groups))):
        return

    dest = split.out_port(0).get_destinations()[0].node
    # The transformation is applicable, only if next node is Concat.
    if dest.soft_get('type') != 'Concat':
        return

    # The transformation is applicable, only if Split is a unique producer for Concat.
    dest_inputs = [p.get_source().node for p in dest.in_ports().values() if not p.disconnected()]
    names_of_concat_inputs = set([n.soft_get('name', n.id) for n in dest_inputs])
    expected_number_of_unique_inputs = 1 if dest.has_valid('axis') else 2
    if len(names_of_concat_inputs) != expected_number_of_unique_inputs:
        return

    return dest


def get_interpolate_pattern(split: Node) -> dict:
    split_shape = split.in_port(0).data.get_shape()
    if len(split_shape) not in {4, 5}:
        return {}
    concat = get_concat_after_split(split)
    if concat is None:
        return {}
    return {'split': split, 'concat': concat}


def get_split_scale(split: Node) -> int:
    split_dests = [d.node for _, p in split.out_ports().items() for d in p.get_connection().get_destinations()]
    num_of_split_dests = len(split_dests)
    num_of_split_out_ports = len(split.out_ports())
    fractional_part = num_of_split_dests / num_of_split_out_ports - num_of_split_dests // num_of_split_out_ports
    assert fractional_part == 0, "Number of output ports of Split must be multiple of number of inputs of Concat"
    return len(split_dests) // len(split.out_ports())


def replace_interpolate_pattern(graph: Graph, match: dict):
    split = match['split']
    scale = float32_array([get_split_scale(split)])
    axis = int(split.in_port(1).get_connection().get_source().node.value)
    split_node_name = split.name
    axis_node = Const(graph, {'name': split_node_name + '/axis', 'value': int64_array([axis])}).create_node()

    shape_node = Shape(graph, dict(name=split_node_name + '/Shape')).create_node()
    scales_node = Const(graph, dict(name=split_node_name + '/scales', value=scale)).create_node()
    mul_node = Mul(graph, dict(name=split_node_name + '/Mul')).create_node()
    scales_node.out_port(0).connect(mul_node.in_port(1))

    strided_slice_node = create_op_with_const_inputs(graph,
                                                     StridedSlice,
                                                     {1: int64_array([axis]), 2: int64_array([axis + 1])},
                                                     {
                                                        'name': split_node_name + '/StridedSlice',
                                                        'begin_mask': int64_array([1]),
                                                        'end_mask': int64_array([1]),
                                                        'new_axis_mask': int64_array([0]),
                                                        'shrink_axis_mask': int64_array([0]),
                                                        'ellipsis_mask': int64_array([0])
                                                     })
    shape_node.out_port(0).connect(strided_slice_node.in_port(0))

    cast_shape_to_float = Cast(graph, {'dst_type': np.float32}).create_node()

    strided_slice_node.out_port(0).connect(cast_shape_to_float.in_port(0))
    cast_shape_to_float.out_port(0).connect(mul_node.in_port(0))

    interp_node = Interpolate(graph,
                              dict(name=split_node_name + '/Interpolate',
                                   mode='nearest',
                                   antialias=0, pads_begin=int64_array([0]), pads_end=int64_array([0]),
                                   coordinate_transformation_mode='half_pixel', nearest_mode='round_prefer_floor',
                                   cube_coeff=-0.75, version='opset4', shape_calculation_mode='scales',
                                   in_ports_count=4, maybe_part_of_sequence=True)).create_node()

    floor_node = Floor(graph, {'name': split_node_name + '/Floor'}).create_node()
    cast_mul_result_to_int = Cast(graph, {'dst_type': np.int64}).create_node()

    mul_node.out_port(0).connect(floor_node.in_port(0))
    floor_node.out_port(0).connect(cast_mul_result_to_int.in_port(0))

    cast_mul_result_to_int.out_port(0).connect(interp_node.in_port(1))
    scales_node.out_port(0).connect(interp_node.in_port(2))
    axis_node.out_port(0).connect(interp_node.in_port(3))

    match['concat'].out_port(0).get_connection().set_source(interp_node.out_port(0))

    split_connection = split.in_port(0).get_connection()
    split_connection.set_destination(interp_node.in_port(0))
    split_connection.get_source().connect(shape_node.in_port(0))


class SplitConcatPairToInterpolate(MiddleReplacementPattern):
    """
    This transformation looks for Interpolation layer implemented using simple operations, i.e. Split and Concat,
    and replaces found pattern with a sequence of Shape, StridedSlice, Const, Mul, Interpolate.

    Found pattern:
        nodes=[
            ('split', dict(kind='op', op='Split')),
            ('concat', dict(kind='op', op='Concat')),
        ],
        edges=[
            ('split', 'concat'),
        ]

    Here we assume that
        1) 'split' is in NDHWC layout and is a 5D-tensor;
        2) split dimensions for 'split' belongs to {1, 2, 3};
        3) all outputs of 'split' go to only inputs of 'concat';
        4) 'concat' takes inputs only from 'split';
        5) split_dim of 'split' is equal to axis of 'concat'.

    Found pattern will be replaced with
        nodes=[
            ('shape', dict(kind='op', op='Shape')),
            ('strided_slice', dict(kind='op', op='StridedSlice')),
            ('scales', dict(kind='op', op='Const')),
            ('scaled_shape', dict(kind='op', op='Mul')),
            ('interp', dict(kind='op', op='Interpolate'))
        ],
        edges=[
            ('shape', 'strided_slice', {'in': 0}),
            ('strided_slice', 'scaled_shape', {'in': 0}),
            ('scales', 'scaled_shape', {'in': 1}),
            ('scaled_shape', 'interp', {'in': 1}),
        ]

    Here scaling factor in Interpolate is equal to a quotient of dividing number of input ports of 'concat'
    by number of output ports of 'split'.
    """
    enabled = True
    force_clean_up = True

    def run_before(self):
        from openvino.tools.mo.middle.InterpolateSequenceToInterpolate import InterpolateSequenceToInterpolate
        return [InterpolateSequenceToInterpolate]

    def find_and_replace_pattern(self, graph: Graph):
        log.debug('Enabled replacement of a pair of Split and Concat with Interpolate.')
        splits = graph.get_op_nodes(op='Split')
        patterns = []

        for split_node in splits:
            interpolate_pattern = get_interpolate_pattern(split_node)
            if interpolate_pattern:
                patterns.append(interpolate_pattern)

        for pattern in patterns:
            replace_interpolate_pattern(graph, pattern)
