"""
 Copyright (c) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging as log
from typing import Optional

from extensions.ops.elementwise import Mul
from extensions.ops.interpolate import Interpolate
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, Node
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
from mo.ops.shape import Shape
from mo.ops.strided_slice import StridedSlice


def get_concat_after_split(split: Node) -> Optional[Node]:
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
    return dest if dest.soft_get('type') == 'Concat' else None


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
    scale = int64_array([get_split_scale(split)])
    axis = int(split.in_port(1).get_connection().get_source().node.value)
    split_node_name = split.name
    axis_node = Const(graph, {'name': split_node_name + '/axis_', 'value': int64_array([axis])}).create_node()

    shape_node = Shape(graph, dict(name=split_node_name + '/Shape_')).create_node()
    scales_node = Const(graph, dict(name=split_node_name + '/scales_', value=scale)).create_node()
    mul_node = Mul(graph, dict(name=split_node_name + '/Mul_')).create_node()
    scales_node.out_port(0).connect(mul_node.in_port(1))

    strided_slice_node = create_op_with_const_inputs(graph,
                                                     StridedSlice,
                                                     {1: int64_array([axis]), 2: int64_array([axis + 1])},
                                                     {
                                                        'name': split_node_name + '/StridedSlice_',
                                                        'begin_mask': int64_array([1]),
                                                        'end_mask': int64_array([1]),
                                                        'new_axis_mask': int64_array([0]),
                                                        'shrink_axis_mask': int64_array([0]),
                                                        'ellipsis_mask': int64_array([0])
                                                     })
    shape_node.out_port(0).connect(strided_slice_node.in_port(0))

    strided_slice_node.out_port(0).connect(mul_node.in_port(0))

    interp_node = Interpolate(graph,
                              dict(name=split_node_name + '/Interpolate_',
                                   mode='nearest',
                                   antialias=0, pads_begin=int64_array([0]), pads_end=int64_array([0]),
                                   coordinate_transformation_mode='half_pixel', nearest_mode='round_prefer_floor',
                                   cube_coeff=-0.75, version='opset4',
                                   in_ports_count=3)).create_node()
    mul_node.out_port(0).connect(interp_node.in_port(1))
    axis_node.out_port(0).connect(interp_node.in_port(2))

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
        from extensions.middle.InterpolateSequenceToInterpolate import InterpolateSequenceToInterpolate
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
