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
import numpy as np

from extensions.ops.activation_ops import Floor
from extensions.ops.Cast import Cast
from extensions.ops.elementwise import Mul
from extensions.ops.interpolate import Interpolate
from mo.front.common.partial_infer.utils import int64_array, float32_array
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, rename_nodes
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.reshape import Reshape
from mo.ops.shape import Shape
from mo.ops.strided_slice import StridedSlice


class UnsqueezeTileReshapeBlockToInterpolate(MiddleReplacementPattern):
    """
    This transformation looks for Interpolation layer implemented using simple operations, i.e. Unsqueeze,
    Tile, Reshape, and replaces found pattern with a sequence of Shape, StridedSlice, Mul, Interpolate.

    Found pattern
        'something' -> Unsqueeze -> Tile -> Reshape
    will be replaced with
        nodes=[
            ('shape', dict(kind='op', op='Shape')),
            ('strided_slice', dict(kind='op', op='StridedSlice')),
            ('to_float', dict(kind='op', op='StridedSlice', dst_type=np.float32)),
            ('m_scales', dict(kind='op', op='Const')),
            ('scaled_shape', dict(kind='op', op='Mul')),
            ('floor', dict(kind='op', op='Floor')),
            ('to_int', dict(kind='op', op='StridedSlice', dst_type=np.float32)),
            ('scales', dict(kind='op', op='Const')),
            ('axes', dict(kind='op', op='Const')),
            ('interp', dict(kind='op', op='Interpolate'))
        ],
        edges=[
            ('something', 'interp', {'in': 0}),
            ('something', 'shape', {'in': 0}),
            ('shape', 'strided_slice', {'in': 0}),
            ('strided_slice', 'to_float', {'in': 0}),
            ('to_float', 'scaled_shape', {'in': 0}),
            ('m_scales', 'scaled_shape', {'in': 1}),
            ('scaled_shape', 'floor'),
            ('floor', 'to_int'),
            ('to_int', 'interp', {'in': 1}),
            ('scales', 'interp', {'in': 2}),
            ('axes', 'interp', {'in': 3}),
        ]
    """
    enabled = True
    force_shape_inference = True

    def run_before(self):
        from extensions.middle.InterpolateSequenceToInterpolate import InterpolateSequenceToInterpolate
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

    @staticmethod
    def is_applicable(match: dict) -> bool:
        unsqueeze_node = match['unsqueeze']
        second_input_of_unsqueeze = unsqueeze_node.in_port(1).get_connection().get_source().node
        if not second_input_of_unsqueeze.has_valid('value') or len(second_input_of_unsqueeze.value) != 1:
            return False

        d_idx = int(second_input_of_unsqueeze.value)

        second_input_of_tile = match['tile'].in_port(1).get_connection().get_source().node
        if not second_input_of_tile.has_valid('value'):
            return False

        input_shape_of_unsqueeze = unsqueeze_node.in_port(0).data.get_shape()
        input_rank_of_unsqueeze = len(input_shape_of_unsqueeze)
        if input_rank_of_unsqueeze not in {4, 5}:
            return False

        if input_rank_of_unsqueeze + 1 != len(second_input_of_tile.value):
            return False

        expected_tile_constant = np.ones(input_rank_of_unsqueeze + 1)
        expected_tile_constant[d_idx] = float(second_input_of_tile.value[d_idx])

        if not np.array_equal(expected_tile_constant, float32_array(second_input_of_tile.value)):
            return False

        reshape_node = match['reshape']
        new_shape = reshape_node.in_port(1).data.get_value()
        if new_shape is None or input_rank_of_unsqueeze != len(new_shape):
            return False

        return True

    def replace_pattern(self, graph: Graph, match: dict):
        if not self.is_applicable(match):
            return

        unsqueeze_node = match['unsqueeze']
        unsqueeze_name = unsqueeze_node.soft_get('name', unsqueeze_node.id)
        second_input_of_unsqueeze = unsqueeze_node.in_port(1).get_connection().get_source().node
        d_idx = int(second_input_of_unsqueeze.value)
        axis = d_idx - 1

        shape_node = Shape(graph, dict(name=unsqueeze_name + '/Shape')).create_node()
        strided_slice_node = create_op_with_const_inputs(graph,
                                                         StridedSlice,
                                                         {
                                                             1: int64_array([axis]),
                                                             2: int64_array([axis + 1])
                                                         },
                                                         {
                                                             'name': unsqueeze_name + '/StridedSlice',
                                                             'begin_mask': int64_array([1]),
                                                             'end_mask': int64_array([1]),
                                                             'new_axis_mask': int64_array([0]),
                                                             'shrink_axis_mask': int64_array([0]),
                                                             'ellipsis_mask': int64_array([0]),
                                                         })
        shape_node.out_port(0).connect(strided_slice_node.in_port(0))
        cast_shape_to_float = Cast(graph, {'dst_type': np.float32}).create_node()
        strided_slice_node.out_port(0).connect(cast_shape_to_float.in_port(0))

        second_input_of_tile = match['tile'].in_port(1).get_connection().get_source().node
        scale = float32_array([second_input_of_tile.value[d_idx]])
        mul_node = create_op_with_const_inputs(graph, Mul, {1: scale}, {'name': unsqueeze_name + '/Mul'})

        cast_shape_to_float.out_port(0).connect(mul_node.in_port(0))
        floor_node = Floor(graph, {'name': unsqueeze_name + '/Floor'}).create_node()
        cast_mul_result_to_int = Cast(graph, {'dst_type': np.int64}).create_node()
        mul_node.out_port(0).connect(floor_node.in_port(0))
        floor_node.out_port(0).connect(cast_mul_result_to_int.in_port(0))

        interp_node = create_op_with_const_inputs(graph,
                                                  Interpolate,
                                                  {
                                                      2: scale,
                                                      3: int64_array([axis])},
                                                  {
                                                      'mode': 'nearest',
                                                      'antialias': 0,
                                                      'pads_begin': int64_array([0]),
                                                      'pads_end': int64_array([0]),
                                                      'coordinate_transformation_mode': 'half_pixel',
                                                      'nearest_mode': 'round_prefer_floor',
                                                      'cube_coeff': -0.75,
                                                      'version': 'opset4',
                                                      'shape_calculation_mode': 'scales',
                                                      'in_ports_count': 4,
                                                      'maybe_part_of_sequence': True
                                                  })
        cast_mul_result_to_int.out_port(0).connect(interp_node.in_port(1))

        reshape_node = match['reshape']
        reshape_node.out_port(0).get_connection().set_source(interp_node.out_port(0))
        reshape_name = reshape_node.soft_get('name', reshape_node.id)
        rename_nodes([(reshape_node, reshape_name + '/delete'), (interp_node, reshape_name)])

        unsqueeze_connection = unsqueeze_node.in_port(0).get_connection()
        unsqueeze_connection.set_destination(interp_node.in_port(0))
        unsqueeze_connection.get_source().connect(shape_node.in_port(0))
