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

from extensions.front.div import Div
from extensions.ops.elementwise import Mul
from extensions.ops.interpolate import Interpolate
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph
from mo.ops.const import Const


class ResizeToInterpolate2D(FrontReplacementSubgraph):
    enabled = True

    def run_before(self):
        return [Div]

    def pattern(self):
        return dict(
            nodes=[
                ('input', dict()),
                ('shape_1', dict(op='ShapeOf')),
                ('shape_2', dict(op='ShapeOf')),
                ('shape_3', dict(op='ShapeOf')),
                ('gather_1', dict(type='Gather')),
                ('gather_2', dict(type='Gather')),
                ('mul_1', dict(op='Mul')),
                ('mul_2', dict(op='Mul')),
                ('unsqueeze_1', dict(op='ExpandDims')),
                ('unsqueeze_2', dict(op='ExpandDims')),
                ('slice', dict(op='Slice')),
                ('slice_start', dict(op='Const', value=lambda x: x is not None and np.array_equal(x, int64_array([2])))),
                ('slice_end', dict(op='Const', value=lambda x: x is not None and np.array_equal(x, int64_array([4])))),
                ('concat_1', dict(op='Concat')),
                ('cast_1', dict(op='Cast')),
                ('cast_2', dict(op='Cast')),
                ('div', dict(op='Div')),
                ('concat_2', dict(op='Concat')),
                ('resize', dict(op='Upsample')),
            ],
            edges=[
                ('input', 'resize', {'in': 0}),
                ('input', 'shape_1', {'in': 0}),
                ('input', 'shape_2', {'in': 0}),
                ('input', 'shape_3', {'in': 0}),
                ('shape_1', 'gather_1', {'in': 0}),
                ('shape_2', 'gather_2', {'in': 0}),
                ('shape_3', 'slice', {'in': 0}),
                ('slice_start', 'slice', {'in': 1}),
                ('slice_end', 'slice', {'in': 2}),
                ('gather_1', 'mul_1', {'in': 0}),
                ('gather_2', 'mul_2', {'in': 0}),
                ('mul_1', 'unsqueeze_1', {'in': 0}),
                ('mul_2', 'unsqueeze_2', {'in': 0}),
                ('unsqueeze_1', 'concat_1', {'in': 0}),
                ('unsqueeze_2', 'concat_1', {'in': 1}),
                ('concat_1', 'cast_1', {'in': 0}),
                ('slice', 'cast_2', {'in': 0}),
                ('cast_1', 'div', {'in': 0}),
                ('cast_2', 'div', {'in': 1}),
                ('div', 'concat_2', {'in': 1}),
                ('concat_2', 'resize', {'in': 1}),
            ])

    def replace_sub_graph(self, graph: Graph, match: dict):
        resize_node = match['resize']

        if match['mul_1'].in_node(1).value != match['mul_2'].in_node(1).value:
            log.info('Pattern matched around resize op {} has different scale values.'.format(resize_node.name))
            return

        interpolate_node = Interpolate(graph, {'name': resize_node.name + '/Interpolate',
                                               'mode': resize_node.mode, 'axes': int64_array([2, 3])}).create_node()

        scale = match['mul_1'].in_node(1).value
        scale_value = int64_array([scale, scale])
        scale_const = Const(graph, {'value': scale_value, 'name': resize_node.name + '/Scale'}).create_node()

        interpolated_shape = Mul(graph, {'name': resize_node.name + '/OutputShape'}).create_node()
        match['slice'].out_port(0).connect(interpolated_shape.in_port(0))
        scale_const.out_port(0).connect(interpolated_shape.in_port(1))

        resize_node.in_port(0).get_connection().set_destination(interpolate_node.in_port(0))
        interpolated_shape.out_port(0).connect(interpolate_node.in_port(1))
        resize_node.out_port(0).get_connection().set_source(interpolate_node.out_port(0))


class ResizeToInterpolate3D(FrontReplacementSubgraph):
    enabled = True

    def run_before(self):
        return [Div]

    def pattern(self):
        return dict(
            nodes=[
                ('input', dict()),
                ('shape_1', dict(op='ShapeOf')),
                ('shape_2', dict(op='ShapeOf')),
                ('shape_3', dict(op='ShapeOf')),
                ('shape_4', dict(op='ShapeOf')),
                ('gather_1', dict(type='Gather')),
                ('gather_2', dict(type='Gather')),
                ('gather_3', dict(type='Gather')),
                ('mul_1', dict(op='Mul')),
                ('mul_2', dict(op='Mul')),
                ('mul_3', dict(op='Mul')),
                ('cast_1', dict(op='Cast')),
                ('cast_2', dict(op='Cast')),
                ('cast_3', dict(op='Cast')),
                ('unsqueeze_1', dict(op='ExpandDims')),
                ('unsqueeze_2', dict(op='ExpandDims')),
                ('unsqueeze_3', dict(op='ExpandDims')),
                ('floor_1', dict(op='Floor')),
                ('floor_2', dict(op='Floor')),
                ('floor_3', dict(op='Floor')),
                ('slice', dict(op='Slice')),
                ('slice_start', dict(op='Const', value=lambda x: x is not None and np.array_equal(x, int64_array([2])))),
                ('slice_end', dict(op='Const', value=lambda x: x is not None and np.array_equal(x, int64_array([5])))),
                ('concat_1', dict(op='Concat')),
                ('cast_4', dict(op='Cast')),
                ('cast_5', dict(op='Cast')),
                ('div', dict(op='Div')),
                ('concat_2', dict(op='Concat')),
                ('resize', dict(op='Upsample')),
            ],
            edges=[
                ('input', 'resize', {'in': 0}),
                ('input', 'shape_1', {'in': 0}),
                ('input', 'shape_2', {'in': 0}),
                ('input', 'shape_3', {'in': 0}),
                ('input', 'shape_4', {'in': 0}),
                ('shape_1', 'gather_1', {'in': 0}),
                ('shape_2', 'gather_2', {'in': 0}),
                ('shape_3', 'gather_3', {'in': 0}),
                ('shape_4', 'slice', {'in': 0}),
                ('slice_start', 'slice', {'in': 1}),
                ('slice_end', 'slice', {'in': 2}),
                ('gather_1', 'mul_1', {'in': 0}),
                ('gather_2', 'mul_2', {'in': 0}),
                ('gather_3', 'mul_3', {'in': 0}),
                ('mul_1', 'cast_1', {'in': 0}),
                ('mul_2', 'cast_2', {'in': 0}),
                ('mul_3', 'cast_3', {'in': 0}),
                ('cast_1', 'floor_1', {'in': 0}),
                ('cast_2', 'floor_2', {'in': 0}),
                ('cast_3', 'floor_3', {'in': 0}),
                ('floor_1', 'unsqueeze_1', {'in': 0}),
                ('floor_2', 'unsqueeze_2', {'in': 0}),
                ('floor_3', 'unsqueeze_3', {'in': 0}),
                ('unsqueeze_1', 'concat_1', {'in': 0}),
                ('unsqueeze_2', 'concat_1', {'in': 1}),
                ('unsqueeze_3', 'concat_1', {'in': 2}),
                ('concat_1', 'cast_4', {'in': 0}),
                ('slice', 'cast_5', {'in': 0}),
                ('cast_4', 'div', {'in': 0}),
                ('cast_5', 'div', {'in': 1}),
                ('div', 'concat_2', {'in': 1}),
                ('concat_2', 'resize', {'in': 1}),
            ])

    def replace_sub_graph(self, graph: Graph, match: dict):
        resize_node = match['resize']
        if match['mul_1'].in_node(1).value != match['mul_2'].in_node(1).value or \
                match['mul_1'].in_node(1).value != match['mul_3'].in_node(1).value:
            log.info('Pattern matched around resize op {} has different scale values.'.format(resize_node.name))
            return

        interpolate_node = Interpolate(graph, {'name': resize_node.name + '/Interpolate',
                                               'mode': resize_node.mode, 'axes': int64_array([2, 3, 4])}).create_node()

        scale = match['mul_1'].in_node(1).value
        scale_value = int64_array([scale, scale, scale])
        scale_const = Const(graph, {'value': scale_value, 'name': resize_node.name + '/Scale'}).create_node()

        interpolated_shape = Mul(graph, {'name': resize_node.name + '/OutputShape'}).create_node()
        match['slice'].out_port(0).connect(interpolated_shape.in_port(0))
        scale_const.out_port(0).connect(interpolated_shape.in_port(1))

        resize_node.in_port(0).get_connection().set_destination(interpolate_node.in_port(0))
        interpolated_shape.out_port(0).connect(interpolate_node.in_port(1))
        resize_node.out_port(0).get_connection().set_source(interpolate_node.out_port(0))
