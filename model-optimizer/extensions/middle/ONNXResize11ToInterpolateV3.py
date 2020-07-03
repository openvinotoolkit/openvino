"""
 Copyright (C) 2020 Intel Corporation

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

from extensions.ops.elementwise import Mul
from extensions.ops.interpolate import Interpolate
from mo.front.common.layout import get_height_dim, get_width_dim, get_depth_dim
from mo.front.common.partial_infer.utils import int64_array
from mo.middle.replacement import MiddleReplacementPattern
from mo.graph.graph import Graph, rename_nodes
from mo.ops.const import Const
from mo.ops.shape import Shape
from mo.ops.strided_slice import StridedSlice


class ONNXResize11ToInterpolate4(MiddleReplacementPattern):
    """
    The transformation replaces ONNX Resize 11 with Interpolate-3.
    """
    enabled = True

    def run_before(self):
        from extensions.middle.InterpolateSequenceToInterpolate import InterpolateSequenceToInterpolate
        return [InterpolateSequenceToInterpolate]

    def pattern(self):
        return dict(
            nodes=[('op', dict(op='ONNXResize11'))],
            edges=[])

    def replace_pattern(self, graph: Graph, match: dict):
        log.debug("Converting of ONNX Resize-11 to Interpolate-3 is triggered.")
        resize = match['op']

        input_shape = resize.in_port(0).data.get_shape()
        input_rank = len(input_shape)
        if input_rank not in {4, 5}:
            log.warning('The input shape is not 4D or 5D for op with name {}'.format(resize.name))
            return

        num_of_inputs = len(resize.in_ports())
        resize_name = resize.name
        assert num_of_inputs in {3, 4}, \
            "Number of inputs of ONNXResize (with name {}) should be equal to 3 or 4".format(resize_name)

        assert resize.soft_get('coordinate_transformation_mode') != 'tf_crop_and_resize', \
            'Mode tf_crop_and_resize is not supported for op {} with name {}'.format(resize.op, resize.name)

        layout = graph.graph['layout']

        if input_rank == 4:
            begin_dim = get_height_dim(layout, input_rank)
            end_dim = get_width_dim(layout, input_rank) + 1
        else:
            begin_dim = get_depth_dim(layout, input_rank)
            end_dim = get_width_dim(layout, input_rank) + 1

        begin_slice = Const(graph, {'name': resize_name + '/begin_slice_',
                                    'value': int64_array([begin_dim])}).create_node()
        end_slice = Const(graph, {'name': resize_name + '/end_slice_',
                                  'value': int64_array([end_dim])}).create_node()
        strided_slice = StridedSlice(graph,
                                     {'name': resize_name + '/StridedSlice_',
                                      'begin_mask': int64_array([1]),
                                      'end_mask': int64_array([1]),
                                      'new_axis_mask': int64_array([0]),
                                      'shrink_axis_mask': int64_array([0]),
                                      'ellipsis_mask': int64_array([0]),
                                      }).create_node()

        begin_slice.out_port(0).connect(strided_slice.in_port(1))
        end_slice.out_port(0).connect(strided_slice.in_port(2))

        interpolate_node = Interpolate(graph, {'version': 'opset4',
                                               'mode': convert_mode(resize.mode),
                                               'coordinate_transformation_mode': resize.coordinate_transformation_mode,
                                               'cube_coeff': resize.cubic_coeff_a,
                                               'nearest_mode': resize.nearest_mode,
                                               'pads_begin': int64_array([0]),
                                               'pads_end': int64_array([0]),
                                               'antialias': 0,
                                               'axes': np.arange(begin_dim, end_dim)}).create_node()

        strided_slice.out_port(0).connect(interpolate_node.in_port(1))

        connection_of_resize_input = resize.in_port(0).get_connection()
        connection_of_resize_input.set_destination(interpolate_node.in_port(0))

        if num_of_inputs == 3:
            shape_of = Shape(graph, {'name': resize_name + '/ShapeOf_'}).create_node()
            mul_node = Mul(graph, {'name': resize_name + '/Mul_'}).create_node()

            shape_of.out_port(0).connect(mul_node.in_port(0))
            mul_node.out_port(0).connect(strided_slice.in_port(0))

            connection_of_resize_input.get_source().connect(shape_of.in_port(0))

            connection_of_scales = resize.in_port(2).get_connection()
            connection_of_scales.set_destination(mul_node.in_port(1))
        else:
            connection_of_sizes = resize.in_port(3).get_connection()
            connection_of_sizes.set_destination(strided_slice.in_port(0))

        rename_nodes([(resize, resize_name + '/delete'), (interpolate_node, resize_name)])

        resize.out_port(0).get_connection().set_source(interpolate_node.out_port(0))


def convert_mode(onnx_mode: str) -> str:
    return {'nearest': 'nearest', 'linear': 'linear_onnx', 'cubic': 'cubic'}[onnx_mode]
