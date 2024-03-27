# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import math
from typing import Dict

import numpy as np

from openvino.tools.mo.ops.Cast import Cast
from openvino.tools.mo.ops.elementwise import Mul
from openvino.tools.mo.ops.interpolate import Interpolate
from openvino.tools.mo.front.common.layout import get_height_dim, get_width_dim, get_depth_dim
from openvino.tools.mo.front.common.partial_infer.utils import int64_array, float32_array
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs, create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph, Node, rename_nodes
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.shape import Shape
from openvino.tools.mo.ops.strided_slice import StridedSlice


class UpsampleToResample(MiddleReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_after(self):
        from openvino.tools.mo.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def run_before(self):
        from openvino.tools.mo.middle.pass_separator import MiddleFinish
        return [MiddleFinish]

    def pattern(self):
        return dict(
            nodes=[
                ('upsample', dict(kind='op', op='Upsample')),
                ('output', dict(kind='data'))],
            edges=[('upsample', 'output')]
        )

    def replace_pattern(self, graph: Graph, match: Dict[str, Node]):
        log.debug('UpsampleToResample is triggered')
        upsample = match['upsample']
        upsample_name = upsample.soft_get('name', upsample.id)
        input_shape = upsample.in_port(0).data.get_shape()
        input_shape_rank = len(input_shape)
        if input_shape_rank not in [4, 5]:
            log.warning('The input shape is not 4D or 5D for op {}'.format(upsample.soft_get('name')))
            return

        depth_scale = None
        layout = graph.graph['layout']

        if len(upsample.in_nodes()) == 2:
            if upsample.in_node(1).value is None:
                return
            scales = upsample.in_node(1).value
            assert len(scales) in (4, 5), 'Supported scales rank is 4 or 5, but it is {} for node {}'.format(
                len(scales), upsample_name)
            if not (math.isclose(scales[0], 1, rel_tol=1e-5) and math.isclose(scales[1], 1, rel_tol=1e-5)):
                return
            height_scale = scales[get_height_dim(layout, input_shape_rank)]
            width_scale = scales[get_width_dim(layout, input_shape_rank)]
            if len(scales) == 5:
                depth_scale = scales[get_depth_dim(layout, input_shape_rank)]
        else:
            height_scale = upsample['height_scale']
            width_scale = upsample['width_scale']

        if 1 in upsample.in_ports() and not upsample.in_port(1).disconnected():
            upsample.in_port(1).disconnect()

        upsample_name = upsample.soft_get('name', upsample.id)
        shape = Shape(graph, {'name': upsample_name + '/0_port'}).create_node()

        layout = graph.graph['layout']

        if input_shape_rank == 4:
            begin_value = int64_array([get_height_dim(layout, input_shape_rank)])
            factor_value = float32_array([height_scale, width_scale])
        else:
            begin_value = int64_array([get_depth_dim(layout, input_shape_rank)])
            factor_value = float32_array([depth_scale, height_scale, width_scale])

        ss = create_op_with_const_inputs(graph, StridedSlice,
                                         {1: begin_value,
                                          2: int64_array([get_width_dim(layout, input_shape_rank) + 1]),
                                          3: int64_array([1])
                                          },
                                         {'name': upsample_name + '/ss_0_port',
                                          'begin_mask': int64_array([1]),
                                          'end_mask': int64_array([1]),
                                          'new_axis_mask': int64_array([0]),
                                          'shrink_axis_mask': int64_array([0]),
                                          'ellipsis_mask': int64_array([0])
                                          })

        mul = create_op_node_with_second_input(graph, Mul, factor_value, {'name': upsample_name + '/factor_mul'})

        source = upsample.in_port(0).get_connection().get_source()
        source.connect(shape.in_port(0))
        shape.out_port(0).connect(ss.in_port(0))

        ss.out_port(0).connect(mul.in_port(0))

        # Create Interpolate operation
        if input_shape_rank == 4:
            axes = int64_array([get_height_dim(layout, input_shape_rank),
                                get_width_dim(layout, input_shape_rank)])
        else:
            axes = int64_array([get_depth_dim(layout, input_shape_rank),
                                get_height_dim(layout, input_shape_rank),
                                get_width_dim(layout, input_shape_rank)])

        axes_node = Const(graph, {'name': upsample_name + '/axis', 'value': axes}).create_node()

        interpolate = Interpolate(graph, {'mode': upsample.attrs()['mode'], 'antialias': 0,
                                          'pads_begin': int64_array([0]), 'pads_end': int64_array([0]),
                                          'coordinate_transformation_mode': 'half_pixel',
                                          'nearest_mode': 'round_prefer_floor', 'cube_coeff': -0.75,
                                          'shape_calculation_mode': 'scales',
                                          'version': 'opset4', 'in_ports_count': 4}).create_node()

        upsample.add_input_port(1, skip_if_exist=True)
        assert upsample.in_port(1).disconnected()
        mul.out_port(0).connect(interpolate.in_port(1))
        axes_node.out_port(0).connect(interpolate.in_port(3))

        scales_node = Const(graph, {'name': upsample_name + '/scales',
                                    'value': factor_value}).create_node()
        scales_node.out_port(0).connect(interpolate.in_port(2))

        upsample.in_port(0).get_connection().set_destination(interpolate.in_port(0))
        upsample.out_port(0).get_connection().set_source(interpolate.out_port(0))

        rename_nodes([(upsample, upsample_name + '/delete'), (interpolate, upsample_name)])

        convert_to_float = Cast(graph, dict(dst_type=np.float32)).create_node()
        convert_to_int = Cast(graph, dict(dst_type=np.int64)).create_node()

        mul.in_port(0).get_connection().insert_node(convert_to_float)
        mul.out_port(0).get_connection().insert_node(convert_to_int)
