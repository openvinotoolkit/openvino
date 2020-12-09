"""
 Copyright (C) 2018-2020 Intel Corporation

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

from extensions.ops.interpolate import Interpolate
from mo.front.common.layout import get_height_dim, get_width_dim
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, Node, rename_nodes
from mo.middle.replacement import MiddleReplacementPattern


def replace_resize_nearest_neighbor(graph: Graph, resize: Node):
    resize_name = resize.soft_get('name', resize.id)
    log.debug("Converting of TF ResizeNearestNeighbor to Interpolate-4 is triggered for node {}.".format(resize_name))

    input_shape = resize.in_port(0).data.get_shape()
    input_rank = len(input_shape)
    if input_rank != 4:
        log.warning('The input shape is not 4D for op with name {}'.format(resize_name))
        return

    num_of_inputs = len([port for port in resize.in_ports().values() if not port.disconnected()])
    assert num_of_inputs == 2, \
        "Number of inputs of TFResizeBilinear (with name {}) should be equal to 2".format(resize_name)

    new_sizes_value = resize.in_port(1).data.get_value()
    assert new_sizes_value is not None, "Node {} with op {} has no value in input port 1".format(resize_name, resize.op)

    attrs_msg = "If half_pixel_centers attribute of the node {} with op {} is True, " \
                "the attribute align_corners must be False"
    assert not resize.half_pixel_centers or (resize.half_pixel_centers and not resize.align_corners), \
        attrs_msg.format(resize_name, resize.op)

    out_height = new_sizes_value[0]
    out_width = new_sizes_value[1]

    layout = graph.graph['layout']
    height_dim = get_height_dim(layout, input_rank)
    width_dim = get_width_dim(layout, input_rank)

    input_height = input_shape[height_dim]
    input_width = input_shape[width_dim]

    align_corners = resize.align_corners
    half_pixel_centers = resize.half_pixel_centers

    nearest_mode = 'floor'
    if align_corners:
        coordinate_transformation_mode = 'align_corners'
        nearest_mode = 'round_prefer_ceil'
    elif half_pixel_centers:
        coordinate_transformation_mode = 'tf_half_pixel_for_nn'
    else:
        coordinate_transformation_mode = 'asymmetric'

    shape_calculation_mode = 'sizes'
    scales_data = np.array([out_height / input_height, out_width / input_width], dtype=np.float32)

    if out_height > 1 and out_width <= 1 and align_corners:
        shape_calculation_mode = 'scales'
        coordinate_transformation_mode = 'asymmetric'
        scales_data = np.array([(out_height - 1) / (input_height - 1), out_width / input_width], dtype=np.float32)
    elif out_height <= 1 and out_width > 1 and align_corners:
        shape_calculation_mode = 'scales'
        coordinate_transformation_mode = 'asymmetric'
        scales_data = np.array([out_height / input_height, (out_width - 1) / (input_width - 1)], dtype=np.float32)

    interpolate4 = create_op_with_const_inputs(graph, Interpolate,
                                               {
                                                   1: int64_array(new_sizes_value),
                                                   2: scales_data,
                                                   3: int64_array([height_dim, width_dim])
                                               },
                                               {
                                                   'name': resize_name + '/interpolate_4',
                                                   'mode': 'nearest',
                                                   'antialias': False,
                                                   'coordinate_transformation_mode': coordinate_transformation_mode,
                                                   'pads_begin': int64_array([0]),
                                                   'pads_end': int64_array([0]),
                                                   'nearest_mode': nearest_mode,
                                                   'cube_coeff': -0.75,
                                                   'shape_calculation_mode': shape_calculation_mode,
                                                   'version': 'opset4',
                                                   'in_ports_count': 4,
                                               })

    input_connection = resize.in_port(0).get_connection()
    input_connection.set_destination(interpolate4.in_port(0))

    resize.out_port(0).get_connection().set_source(interpolate4.out_port(0))
    rename_nodes([(resize, resize_name + '/delete_'), (interpolate4, resize_name)])


class TFResizeNearestNeighborToInterpolateV4(MiddleReplacementPattern):
    """
    The transformation replaces TF ResizeNearestNeighbor with Interpolate-4.
    """
    enabled = True

    def run_before(self):
        from extensions.middle.InterpolateSequenceToInterpolate import InterpolateSequenceToInterpolate
        return [InterpolateSequenceToInterpolate]

    def find_and_replace_pattern(self, graph: Graph):
        resize_nearest_ops = graph.get_op_nodes(op='TFResizeNearestNeighbor')
        for resize in resize_nearest_ops:
            replace_resize_nearest_neighbor(graph, resize)
