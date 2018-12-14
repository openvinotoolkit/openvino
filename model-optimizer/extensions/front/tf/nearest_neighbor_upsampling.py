"""
 Copyright (c) 2018 Intel Corporation

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

import networkx as nx

from extensions.front.tf.Pack import Pack
from extensions.ops.resample import ResampleOp
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import replace_node


class NearestNeighborUpsampling(FrontReplacementSubgraph):
    enabled = True

    def run_before(self):
        return [Pack]

    def pattern(self):
        return dict(
            nodes=[('op', dict(kind='op')),
                   ('shape', dict(kind='op', op='Shape')),
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

    def replace_sub_graph(self, graph: nx.MultiDiGraph, match: dict):
        log.debug('Matched NearestNeighborUpsampling pattern: {}'.format([node.id for node in match.values()]))
        try:
            input_height = match['pack_1'].in_node(1).value.item()
            input_width = match['pack_1'].in_node(3).value.item()

            height_scale = match['mul_const'].shape[-4]
            width_scale = match['mul_const'].shape[-2]
        except Exception as ex:
            log.warning('Failed to determine scaling parameters from the topology. Do not apply pattern.')
            return

        resample_op = ResampleOp(graph, {'width': input_width * width_scale, 'height': input_height * height_scale,
                                         'name': 'Resample_', 'antialias': 0,
                                         'resample_type': 'caffe.ResampleParameter.NEAREST'})
        resample_node = resample_op.create_node([match['op']])

        replace_node(match['reshape_2'], resample_node)
        graph.remove_nodes_from([node.id for node in match.values() if node.id != match['op'].id])
