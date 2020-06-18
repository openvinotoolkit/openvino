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

from extensions.middle.InterpolateSequenceToInterpolate import InterpolateSequenceToInterpolate
from extensions.ops.elementwise import Mul
from extensions.ops.interpolate import Interpolate
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph, rename_nodes
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
from mo.ops.shape import Shape
from mo.ops.strided_slice import StridedSlice


class UnsqueezeTileReshapeBlockToInterpolate(MiddleReplacementPattern):
    """
    This transformation looks for Interpolation layer implemented using simple operations, i.e. Unsqueeze,
    Tile, Reshape, and replaces found pattern with a sequence of Shape, StridedSlice, Const, Mul, Interpolate.

    Here we assume that the input of 'unsqueeze' is in NDHWC layout and is a 5D-tensor.

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
    """
    enabled = True
    force_shape_inference = True

    def run_before(self):
        return [InterpolateSequenceToInterpolate]

    def pattern(self):
        log.debug('Enabled replacement of a sequence of Unsqueeze, Tile, Reshape with Interpolate.')
        return dict(
            nodes=[
                ('unsqueeze', dict(kind='op', op='Unsqueeze')),
                ('unsqueeze_data', dict(kind='data')),
                ('tile', dict(kind='op', op='Tile')),
                ('tile_data', dict(kind='data')),
                ('reshape', dict(kind='op', op='Reshape')),
            ],
            edges=[
                ('unsqueeze', 'unsqueeze_data'),
                ('unsqueeze_data', 'tile', {'in': 0}),
                ('tile', 'tile_data'),
                ('tile_data', 'reshape', {'in': 0}),
            ]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        unsqueeze_node = match['unsqueeze']
        unsqueeze_name = unsqueeze_node.name

        second_input_of_unsqueeze = unsqueeze_node.in_port(1).get_connection().get_source().node
        if not second_input_of_unsqueeze.has_valid('value'):
            return

        d_idx = int(second_input_of_unsqueeze.value)

        second_input_of_tile = match['tile'].in_port(1).get_connection().get_source().node
        if not second_input_of_tile.has_valid('value'):
            return

        input_shape_of_unsqueeze = unsqueeze_node.in_port(0).data.get_shape()
        if len(input_shape_of_unsqueeze) not in {4, 5}:
            return

        scale = int64_array([second_input_of_tile.value[d_idx]])
        axis = d_idx - 1

        shape_node = Shape(graph, dict(name=unsqueeze_name + '/Shape_')).create_node()
        scales_node = Const(graph, dict(name=unsqueeze_name + '/scales_', value=scale)).create_node()
        mul_node = Mul(graph, dict(name=unsqueeze_name + '/Mul_')).create_node()
        scales_node.out_port(0).connect(mul_node.in_port(1))

        slice_begin = Const(graph, dict(name=unsqueeze_name + '/slice_begin_', value=int64_array([axis]))).create_node()
        slice_end = Const(graph, dict(name=unsqueeze_name + '/slice_end_', value=int64_array([axis + 1]))).create_node()

        strided_slice_node = StridedSlice(graph,
                                          {'name': unsqueeze_name + '/StridedSlice_',
                                           'begin_mask': int64_array([1]),
                                           'end_mask': int64_array([1]),
                                           'new_axis_mask': int64_array([0]),
                                           'shrink_axis_mask': int64_array([0]),
                                           'ellipsis_mask': int64_array([0]),
                                           }).create_node()
        shape_node.out_port(0).connect(strided_slice_node.in_port(0))
        slice_begin.out_port(0).connect(strided_slice_node.in_port(1))
        slice_end.out_port(0).connect(strided_slice_node.in_port(2))
        strided_slice_node.out_port(0).connect(mul_node.in_port(0))

        interp_node = Interpolate(graph,
                                  dict(axes=int64_array([axis]), mode='nearest',
                                       antialias=0, pads_begin=int64_array([0]),
                                       pads_end=int64_array([0]), coordinate_transformation_mode='half_pixel',
                                       nearest_mode='round_prefer_floor', cube_coeff=-0.75,
                                       version='opset3')).create_node()
        mul_node.out_port(0).connect(interp_node.in_port(1))

        reshape_node = match['reshape']

        reshape_node.out_port(0).get_connection().set_source(interp_node.out_port(0))
        reshape_name = reshape_node.name
        rename_nodes([(reshape_node, reshape_name + '/delete'), (interp_node, reshape_name)])

        unsqueeze_connection = match['unsqueeze'].in_port(0).get_connection()
        before_unsqueeze = unsqueeze_connection.get_source().node
        unsqueeze_connection.set_destination(interp_node.in_port(0))
        before_unsqueeze.out_port(0).connect(shape_node.in_port(0))
