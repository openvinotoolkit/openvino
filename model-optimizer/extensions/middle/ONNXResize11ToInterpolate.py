# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from extensions.ops.Cast import Cast
from extensions.ops.activation_ops import Floor
from extensions.ops.elementwise import Add, Div, Mul
from extensions.ops.interpolate import Interpolate
from mo.front.common.layout import get_depth_dim, get_height_dim, get_width_dim
from mo.front.common.partial_infer.utils import int64_array, float_array
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, Node, rename_nodes
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
from mo.ops.shape import Shape
from mo.ops.strided_slice import StridedSlice


def convert_mode(onnx_mode: str) -> str:
    return {'nearest': 'nearest', 'linear': 'linear_onnx', 'cubic': 'cubic'}[onnx_mode]


def replace_resize(graph: Graph, resize: Node):
    log.debug("Converting of ONNX Resize-11 to Interpolate-4 "
              "is triggered for node {}.".format(resize.soft_get('name', resize.id)))

    input_shape = resize.in_port(0).data.get_shape()
    input_rank = len(input_shape)
    resize_name = resize.soft_get('name', resize.id)
    if input_rank not in {4, 5}:
        log.warning('The input shape is not 4D or 5D for op with name {}'.format(resize_name))
        return

    assert (resize.is_in_port_connected(0) and (resize.is_in_port_connected(2) or resize.is_in_port_connected(3))), \
        "Scales or sizes inputs must be connected to Node {} with op {}.".format(resize.soft_get("name", resize.id),
                                                                                 resize.op)

    assert resize.soft_get('coordinate_transformation_mode') != 'tf_crop_and_resize', \
        'Mode tf_crop_and_resize is not supported for op {} with name {}'.format(resize.op,
                                                                                 resize.soft_get("name", resize.id))

    layout = graph.graph['layout']

    if input_rank == 4:
        begin_dim = get_height_dim(layout, input_rank)
        end_dim = get_width_dim(layout, input_rank) + 1
    else:
        begin_dim = get_depth_dim(layout, input_rank)
        end_dim = get_width_dim(layout, input_rank) + 1

    sizes_ss = create_op_with_const_inputs(graph, StridedSlice,
                                           {1: int64_array([begin_dim]),
                                            2: int64_array([end_dim]),
                                            3: int64_array([1])},
                                           {'name': resize_name + '/StridedSlice_sizes',
                                            'begin_mask': int64_array([1]),
                                            'end_mask': int64_array([1]),
                                            'new_axis_mask': int64_array([0]),
                                            'shrink_axis_mask': int64_array([0]),
                                            'ellipsis_mask': int64_array([0])})
    scales_ss = create_op_with_const_inputs(graph, StridedSlice,
                                            {1: int64_array([begin_dim]),
                                             2: int64_array([end_dim]),
                                             3: int64_array([1])},
                                            {'name': resize_name + '/StridedSlice_scales',
                                             'begin_mask': int64_array([1]),
                                             'end_mask': int64_array([1]),
                                             'new_axis_mask': int64_array([0]),
                                             'shrink_axis_mask': int64_array([0]),
                                             'ellipsis_mask': int64_array([0])})
    axes_node = Const(graph,
                      {'name': resize_name + '/axis',
                       'value': int64_array(np.arange(begin_dim, end_dim))}).create_node()

    shape_calculation_mode = 'sizes' if resize.is_in_port_connected(3) else 'scales'

    interpolate_node = Interpolate(graph, {'version': 'opset4',
                                           'mode': convert_mode(resize.mode),
                                           'coordinate_transformation_mode': resize.coordinate_transformation_mode,
                                           'cube_coeff': resize.cube_coeff,
                                           'nearest_mode': resize.nearest_mode,
                                           'pads_begin': int64_array([0]),
                                           'pads_end': int64_array([0]),
                                           'antialias': 0,
                                           'shape_calculation_mode': shape_calculation_mode,
                                           'in_ports_count': 4}).create_node()

    axes_node.out_port(0).connect(interpolate_node.in_port(3))
    shape_of = Shape(graph, {'name': resize_name + '/ShapeOf'}).create_node()

    add_node = create_op_with_const_inputs(graph, Add,
                                           {1: float_array([1.0e-5])},
                                           {'name': resize_name + '/Add'})

    dst_dtype = np.float32  # even if data_type=FP16 use float32 for shape values

    if not resize.is_in_port_connected(3):
        cast_shape_to_float = Cast(graph, {'dst_type': dst_dtype}).create_node()
        mul_node = Mul(graph, {'name': resize_name + '/Mul'}).create_node()
        shape_of.out_port(0).connect(cast_shape_to_float.in_port(0))
        cast_shape_to_float.out_port(0).connect(mul_node.in_port(0))
        cast_add_result_to_int = Cast(graph, {'dst_type': np.int64}).create_node()
        floor_node = Floor(graph, {'name': resize_name + '/Floor'}).create_node()
        mul_node.out_port(0).connect(add_node.in_port(0))
        add_node.out_port(0).connect(floor_node.in_port(0))
        floor_node.out_port(0).connect(cast_add_result_to_int.in_port(0))
        cast_add_result_to_int.out_port(0).connect(sizes_ss.in_port(0))
        sizes_ss.out_port(0).connect(interpolate_node.in_port(1))
        scales_ss.out_port(0).connect(interpolate_node.in_port(2))

        connection_of_resize_input = resize.in_port(0).get_connection()
        connection_of_resize_input.set_destination(interpolate_node.in_port(0))

        connection_of_scales = resize.in_port(2).get_connection()
        connection_of_scales.set_destination(scales_ss.in_port(0))

        connection_of_resize_input.get_source().connect(shape_of.in_port(0))
        connection_of_scales.get_source().connect(mul_node.in_port(1))
    else:
        cast_shape_to_float = Cast(graph, {'dst_type': dst_dtype}).create_node()
        cast_sizes_to_float = Cast(graph, {'dst_type': dst_dtype}).create_node()
        div_node = Div(graph, {'name': resize_name + '/Div'}).create_node()
        cast_sizes_to_float.out_port(0).connect(div_node.in_port(0))
        cast_shape_to_float.out_port(0).connect(div_node.in_port(1))
        shape_of.out_port(0).connect(cast_shape_to_float.in_port(0))
        div_node.out_port(0).connect(add_node.in_port(0))
        add_node.out_port(0).connect(scales_ss.in_port(0))
        scales_ss.out_port(0).connect(interpolate_node.in_port(2))
        sizes_ss.out_port(0).connect(interpolate_node.in_port(1))

        connection_of_resize_input = resize.in_port(0).get_connection()
        connection_of_resize_input.set_destination(interpolate_node.in_port(0))

        connection_of_sizes = resize.in_port(3).get_connection()
        connection_of_sizes.set_destination(sizes_ss.in_port(0))

        connection_of_resize_input.get_source().connect(shape_of.in_port(0))
        connection_of_sizes.get_source().connect(cast_sizes_to_float.in_port(0))

    rename_nodes([(resize, resize_name + '/delete'), (interpolate_node, resize_name)])
    resize.out_port(0).get_connection().set_source(interpolate_node.out_port(0))


class ONNXResize11ToInterpolate(MiddleReplacementPattern):
    """
    The transformation replaces ONNX Resize 11 with Interpolate-4.
    """
    enabled = True

    def run_before(self):
        from extensions.middle.InterpolateSequenceToInterpolate import InterpolateSequenceToInterpolate
        return [InterpolateSequenceToInterpolate]

    def find_and_replace_pattern(self, graph: Graph):
        resize11_ops = graph.get_op_nodes(op='ONNXResize11')
        for resize in resize11_ops:
            replace_resize(graph, resize)
