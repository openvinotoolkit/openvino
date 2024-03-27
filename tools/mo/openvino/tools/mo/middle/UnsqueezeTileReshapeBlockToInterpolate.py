# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.ops.elementwise import Mul
from openvino.tools.mo.ops.interpolate import Interpolate
from openvino.tools.mo.front.common.partial_infer.utils import int64_array, float32_array
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, rename_nodes
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.shape import Shape
from openvino.tools.mo.utils.shape import node_to_get_shape_value_of_indices


class UnsqueezeTileReshapeBlockToInterpolate(MiddleReplacementPattern):
    """
    The transformation looks for a sub-graph performing unsqueeze-ing input tensor by some "axis" and then tiling over
    it fixed number of "times". This pattern can be represented with the Interpolate operation  of mode "nearest"
    performing interpolation over specific "axis" with fixed output dimension size equal to "times".

    Note, that the transformation expects that the output from Tile is reshaped back to the tensor with rank equal to
    the input tensor rank. This constraints occurs because the pattern appears in the models where these patterns appear
    one after another, performing unsqueeze-ing over different dimensions, effectively performing interpolation over
    several dimensions.

    These sequences are merged in the 'optimizer/extensions/middle/InterpolateSequenceToInterpolate.py' transformation
    into a single Interpolate operation.

    The transformation is applicable only when all following conditions are fulfilled:

    1. 'Unsqueeze' must be performed with respect to only one axis.
    2. The length of the value of the second input of 'Tile' must be equal to the input rank of 'Unsqueeze' plus 1.
    3. All elements of the value of the second input of 'Tile' must be equal to 1,
       except the value corresponding the interpolated axis.
    4. The input rank of 'Unsqueeze' and the output rank of 'Reshape' must be equal.

    Finally, because plugins support only Interpolate-4 with 4D or 5D tensor with interpolated data,
    we need to check that the input rank of 'Unsqueeze' is equal to 4 or 5.

    Example.

    Let data = np.arange(0, 1 * 2 * 3 * 4).reshape((1, 2, 3, 4)).astype(np.float32), that is
        data = mo_array([[[[ 0,  1,  2,  3],
                           [ 4,  5,  6,  7],
                           [ 8,  9, 10, 11]],
                          [[12, 13, 14, 15],
                           [16, 17, 18, 19],
                           [20, 21, 22, 23]]]], dtype=np.float32)
    After np.tile(np.expand_dims(data, 3), [1, 1, 1, 2, 1]).reshape((1, 2, 3 * 2, 4)) we get
        array([[[[ 0,  1,  2,  3],
                 [ 0,  1,  2,  3],
                 [ 4,  5,  6,  7],
                 [ 4,  5,  6,  7],
                 [ 8,  9, 10, 11],
                 [ 8,  9, 10, 11]],
                [[12, 13, 14, 15],
                 [12, 13, 14, 15],
                 [16, 17, 18, 19],
                 [16, 17, 18, 19],
                 [20, 21, 22, 23],
                 [20, 21, 22, 23]]]], dtype=np.float32)
    This result is equal to nearest interpolation along with axis = 2 (the second argument of 'expand_dims')
    and scale = 2 (the element from the second argument of 'tile' that is not equal to 1).
    """
    enabled = True
    force_shape_inference = True

    def run_before(self):
        from openvino.tools.mo.middle.InterpolateSequenceToInterpolate import InterpolateSequenceToInterpolate
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
        """
        This function checks whether this transformation is applicable.
        :param match: dictionary with nodes from the found pattern
        :return: True, if the transformation is applicable
                 False, otherwise
        """
        unsqueeze_node = match['unsqueeze']
        second_input_of_unsqueeze = unsqueeze_node.in_port(1).get_connection().get_source().node
        if not second_input_of_unsqueeze.has_valid('value') or len(second_input_of_unsqueeze.value) != 1:
            return False

        d_idx = int(second_input_of_unsqueeze.value)
        if d_idx == 0:
            return False

        second_input_of_tile = match['tile'].in_port(1).get_connection().get_source().node
        if not second_input_of_tile.has_valid('value'):
            return False

        input_shape_of_unsqueeze = unsqueeze_node.in_port(0).data.get_shape()
        input_rank_of_unsqueeze = len(input_shape_of_unsqueeze)
        if input_rank_of_unsqueeze not in {4, 5}:
            return False

        if input_rank_of_unsqueeze + 1 != len(second_input_of_tile.value):
            return False

        expected_tile_constant = np.ones(input_rank_of_unsqueeze + 1, dtype=np.float32)
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
        axis_len_node = node_to_get_shape_value_of_indices(shape_node, [axis])

        second_input_of_tile = match['tile'].in_port(1).get_connection().get_source().node
        scale = int64_array([second_input_of_tile.value[d_idx]])
        float_scale = float32_array([second_input_of_tile.value[d_idx]])
        mul_node = create_op_with_const_inputs(graph, Mul, {1: scale}, {'name': unsqueeze_name + '/Mul'})

        axis_len_node.out_port(0).connect(mul_node.in_port(0))

        interp_node = create_op_with_const_inputs(graph,
                                                  Interpolate,
                                                  {
                                                      2: float_scale,
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
        mul_node.out_port(0).connect(interp_node.in_port(1))

        reshape_node = match['reshape']
        reshape_node.out_port(0).get_connection().set_source(interp_node.out_port(0))
        reshape_name = reshape_node.soft_get('name', reshape_node.id)
        rename_nodes([(reshape_node, reshape_name + '/delete'), (interp_node, reshape_name)])

        unsqueeze_connection = unsqueeze_node.in_port(0).get_connection()
        unsqueeze_connection.set_destination(interp_node.in_port(0))
        unsqueeze_connection.get_source().connect(shape_node.in_port(0))
