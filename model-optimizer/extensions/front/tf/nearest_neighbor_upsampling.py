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

from extensions.front.Pack import Pack
from extensions.ops.interpolate import Interpolate
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph
from mo.ops.const import Const


class NearestNeighborUpsampling(FrontReplacementSubgraph):
    enabled = True

    def run_before(self):
        return [Pack]

    def pattern(self):
        return dict(
            nodes=[('op', dict(kind='op')),
                   ('shape', dict(kind='op', op='ShapeOf')),
                   ('strided_slice', dict(kind='op', op='StridedSlice')),
                   ('pack_1', dict(kind='op', op='Pack')),
                   ('reshape_1', dict(kind='op', op='Reshape')),
                   ('mul_const', dict(kind='op', op='Const')),
                   ('mul', dict(kind='op', op='Mul')),
                   ('pack_2', dict(kind='op', op='Pack')),
                   ('reshape_2', dict(kind='op', op='Reshape')),
                   ],
            edges=[
                ('op', 'shape'),
                ('op', 'reshape_1'),
                ('shape', 'strided_slice'),
                ('strided_slice', 'pack_1'),
                ('strided_slice', 'pack_2'),
                ('pack_1', 'reshape_1'),
                ('pack_2', 'reshape_2'),
                ('reshape_1', 'mul'),
                ('mul_const', 'mul'),
                ('mul', 'reshape_2'),
            ]
        )

    def replace_sub_graph(self, graph: Graph, match: dict):
        log.debug('Matched NearestNeighborUpsampling pattern: {}'.format([node.id for node in match.values()]))
        try:
            input_height = match['pack_1'].in_node(1).value.item()
            input_width = match['pack_1'].in_node(3).value.item()

            height_scale = match['mul_const'].shape[-4]
            width_scale = match['mul_const'].shape[-2]
        except Exception as ex:
            log.warning('Failed to determine scaling parameters from the topology. Do not apply pattern.')
            return

        reshape2_name = match['reshape_2'].name
        resample_op = Interpolate(graph,
                                  {'mode': 'nearest', 'antialias': 0, 'pads_begin': int64_array([0]),
                                   'pads_end': int64_array([0]), 'coordinate_transformation_mode': 'half_pixel',
                                   'nearest_mode': 'round_prefer_floor', 'cube_coeff': -0.75, 'version': 'opset4',
                                   'name': reshape2_name + '/Resample_',
                                   'in_ports_count': 3})
        resample_node = resample_op.create_node([match['op']])
        axes_node = Const(graph,
                          {
                              'name': resample_node.name + '/axes_',
                              'value': int64_array([2, 3]) if graph.graph['layout'] == 'NCHW' else int64_array([1, 2])
                          }).create_node()
        const = Const(graph, {'value': np.array([input_height * height_scale, input_width * width_scale]),
                              'name': resample_node.name + '/target_shape'}).create_node()

        match['reshape_2'].replace_node(resample_node)

        resample_node.add_input_port(1, skip_if_exist=True)
        assert resample_node.in_port(1).disconnected()
        const.out_port(0).connect(resample_node.in_port(1))
        axes_node.out_port(0).connect(resample_node.in_port(2))

        graph.remove_nodes_from([node.id for node in match.values() if node.id != match['op'].id])
