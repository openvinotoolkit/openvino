# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.ops.Cast import Cast
from openvino.tools.mo.ops.elementwise import Div
from openvino.tools.mo.ops.interpolate import Interpolate
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, Node, rename_nodes
from openvino.tools.mo.ops.shape import Shape
from openvino.tools.mo.ops.strided_slice import StridedSlice


def replace_tf_resize(graph: Graph, resize: Node, interpolation_mode: str):
    resize_name = resize.soft_get('name', resize.id)
    log.debug("Converting of {} to Interpolate-4 is triggered for node {}.".format(resize.op, resize_name))

    num_of_inputs = len([port for port in resize.in_ports().values() if not port.disconnected()])
    assert num_of_inputs == 2, \
        "Number of inputs of {} (with name {}) should be equal to 2".format(resize.op, resize_name)

    attrs_msg = "If half_pixel_centers attribute of the node {} with op {} is True, " \
                "the attribute align_corners must be False"
    assert not resize.half_pixel_centers or (resize.half_pixel_centers and not resize.align_corners), \
        attrs_msg.format(resize_name, resize.op)

    if resize.has_valid('data_type') and not np.issubdtype(resize.data_type, np.floating):
       input_cast = Cast(graph, {'name': resize_name + '/to_f32', 'dst_type': np.float32}).create_node()
       resize.in_port(0).get_connection().insert_node(input_cast, "source")
       # casted tensor is not present in TF model, we don't need to propagate any name to output of this Convert
       # therefore we use "source" attributes_save_mode for insert_node

    shape = Shape(graph, {'name': resize_name + '/shapeof'}).create_node()

    ss = create_op_with_const_inputs(graph, StridedSlice,
                                     {1: int64_array([1]),
                                      2: int64_array([3]),
                                      3: int64_array([1])
                                      },
                                     {'name': resize_name + '/StridedSlice',
                                      'begin_mask': int64_array([1]),
                                      'end_mask': int64_array([1]),
                                      'new_axis_mask': int64_array([0]),
                                      'shrink_axis_mask': int64_array([0]),
                                      'ellipsis_mask': int64_array([0])
                                      })

    div_node = Div(graph, {'name': resize_name + '/Div'}).create_node()

    shape_to_float = Cast(graph, dict(dst_type=np.float32)).create_node()
    size_to_float = Cast(graph, dict(dst_type=np.float32)).create_node()

    size_to_float.out_port(0).connect(div_node.in_port(0))
    shape_to_float.out_port(0).connect(div_node.in_port(1))
    ss.out_port(0).connect(shape_to_float.in_port(0))
    shape.out_port(0).connect(ss.in_port(0))

    align_corners = resize.align_corners
    half_pixel_centers = resize.half_pixel_centers

    nearest_mode = 'floor' if interpolation_mode == 'nearest' else 'round_prefer_floor'
    if align_corners:
        coordinate_transformation_mode = 'align_corners'
        if interpolation_mode == 'nearest':
            nearest_mode = 'round_prefer_ceil'
    elif half_pixel_centers:
        coordinate_transformation_mode = 'tf_half_pixel_for_nn' if interpolation_mode == 'nearest' else 'half_pixel'
    else:
        coordinate_transformation_mode = 'asymmetric'

    interpolate4 = create_op_with_const_inputs(graph, Interpolate,
                                               {
                                                   3: int64_array([1, 2])
                                               },
                                               {
                                                   'name': resize_name + '/interpolate_4',
                                                   'mode': interpolation_mode,
                                                   'antialias': False,
                                                   'coordinate_transformation_mode': coordinate_transformation_mode,
                                                   'pads_begin': int64_array([0]),
                                                   'pads_end': int64_array([0]),
                                                   'nearest_mode': nearest_mode,
                                                   'cube_coeff': -0.75,
                                                   'shape_calculation_mode': 'sizes',
                                                   'version': 'opset4',
                                                   'in_ports_count': 4,
                                               })

    resize_input_connection = resize.in_port(0).get_connection()
    resize_input_connection.set_destination(interpolate4.in_port(0))
    resize_input_connection.get_source().connect(shape.in_port(0))

    div_node.out_port(0).connect(interpolate4.in_port(2))

    sizes_connection = resize.in_port(1).get_connection()
    sizes_connection.set_destination(interpolate4.in_port(1))
    sizes_connection.get_source().connect(size_to_float.in_port(0))

    resize.out_port(0).get_connection().set_source(interpolate4.out_port(0))
    rename_nodes([(resize, resize_name + '/delete'), (interpolate4, resize_name)])


class TFResizeToInterpolate(FrontReplacementOp):
    """
    The transformation replaces TFResize with Interpolate-4.
    """
    op = 'TFResize'
    enabled = True

    def run_after(self):
        from openvino.tools.mo.front.InterpolateNormalizer import InterpolateNormalizer
        return [InterpolateNormalizer]

    def replace_sub_graph(self, graph: Graph, match: dict):
        resize = match['op']
        replace_tf_resize(graph, resize, resize.mode)
