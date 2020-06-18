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
import math
from typing import Dict

import numpy as np

from extensions.ops.Cast import Cast
from extensions.ops.elementwise import Mul
from extensions.ops.interpolate import Interpolate
from mo.front.common.layout import get_height_dim, get_width_dim, get_depth_dim
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph, Node, rename_nodes
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
from mo.ops.shape import Shape
from mo.ops.strided_slice import StridedSlice


class UpsampleToResample(MiddleReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_after(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def run_before(self):
        from extensions.middle.pass_separator import MiddleFinish
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
        input_shape = upsample.in_port(0).data.get_shape()
        input_shape_rank = len(input_shape)
        if input_shape_rank not in [4, 5]:
            log.warning('The input shape is not 4D or 5D for op {}'.format(upsample.soft_get('name')))
            return

        depth_scale = None
        if len(upsample.in_nodes()) == 2:
            if upsample.in_node(1).value is None:
                return
            scales = upsample.in_node(1).value
            assert len(scales) in (4, 5), 'Supported scales rank is 4 or 5, but it is {} for node {}'.format(
                len(scales), upsample.soft_get('name', upsample.id)
            )
            if not (math.isclose(scales[0], 1, rel_tol=1e-5) and math.isclose(scales[1], 1, rel_tol=1e-5)):
                return
            height_scale = scales[2]
            width_scale = scales[3]
            if len(scales) == 5:
                depth_scale = scales[4]
        else:
            height_scale = upsample['height_scale']
            width_scale = upsample['width_scale']

        if 1 in upsample.in_ports() and not upsample.in_port(1).disconnected():
            upsample.in_port(1).disconnect()

        upsample_name = upsample.name
        shape = Shape(graph, {'name': upsample_name + '/0_port'}).create_node()

        layout = graph.graph['layout']
        if input_shape_rank == 4:
            begin = Const(graph, {'value': int64_array([get_height_dim(layout, input_shape_rank)])}).create_node()
            factor = Const(graph, {'value': np.array([height_scale, width_scale])}).create_node()
        else:
            begin = Const(graph, {'value': int64_array([get_depth_dim(layout, input_shape_rank)])}).create_node()
            factor = Const(graph, {'value': np.array([depth_scale, height_scale, width_scale])}).create_node()
        end = Const(graph, {'value': int64_array([get_width_dim(layout, input_shape_rank) + 1])}).create_node()

        stride = Const(graph, {'value': int64_array([1])}).create_node()
        ss = StridedSlice(graph, {'name': upsample_name + '/ss_0_port',
                                  'begin_mask': int64_array([1]),
                                  'end_mask': int64_array([1]),
                                  'new_axis_mask': int64_array([0]),
                                  'shrink_axis_mask': int64_array([0]),
                                  'ellipsis_mask': int64_array([0])}).create_node()

        mul = Mul(graph, {'name': upsample_name + '/factor_mul_'}).create_node()

        source = upsample.in_port(0).get_connection().get_source()
        source.connect(shape.in_port(0))
        shape.out_port(0).connect(ss.in_port(0))
        begin.out_port(0).connect(ss.in_port(1))
        end.out_port(0).connect(ss.in_port(2))
        stride.out_port(0).connect(ss.in_port(3))
        ss.out_port(0).connect(mul.in_port(0))
        factor.out_port(0).connect(mul.in_port(1))

        # Create Interpolate operation
        if input_shape_rank == 4:
            axes = int64_array([get_height_dim(layout, input_shape_rank),
                                get_width_dim(layout, input_shape_rank)])
        else:
            axes = int64_array([get_depth_dim(layout, input_shape_rank),
                                get_height_dim(layout, input_shape_rank),
                                get_width_dim(layout, input_shape_rank)])

        resample_op = Interpolate(graph, {'axes': axes, 'mode': upsample.attrs()['mode'], 'antialias': 0,
                                          'convert_to_resample': True, 'pads_begin': int64_array([0]),
                                          'pads_end': int64_array([0]), 'coordinate_transformation_mode': 'half_pixel',
                                          'nearest_mode': 'round_prefer_floor', 'cube_coeff': -0.75,
                                          'version': 'opset3'}).create_node()

        upsample.add_input_port(1, skip_if_exist=True)
        assert upsample.in_port(1).disconnected()
        mul.out_port(0).connect(resample_op.in_port(1))

        upsample.in_port(0).get_connection().set_destination(resample_op.in_port(0))
        upsample.out_port(0).get_connection().set_source(resample_op.out_port(0))

        rename_nodes([(upsample, upsample_name + '/delete_'), (resample_op, upsample_name)])

        convert_to_float = Cast(graph, dict(dst_type=np.float32)).create_node()
        int_np_type = np.int64 if graph.graph['cmd_params'].generate_experimental_IR_V10 else np.int32
        convert_to_int = Cast(graph, dict(dst_type=int_np_type)).create_node()

        mul.in_port(0).get_connection().insert_node(convert_to_float)
        mul.out_port(0).get_connection().insert_node(convert_to_int)
