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

from extensions.ops.gather import Gather
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
from mo.ops.op import PermuteAttrs


class Deconvolution3rdInputNormalization(MiddleReplacementPattern):
    enabled = True
    force_clean_up = True

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('op', dict(kind='op', type='Deconvolution'))],
            edges=[]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['op']
        if not node.has_port('in', 2) or node.in_port(2).disconnected() or not node.has_and_set('shape_input'):
            return

        if node.has_valid('layout') and not node.layout.startswith('NC') and graph.graph['layout'] == 'NCHW':
            input_shape_rank = len(node.in_port(0).data.get_shape())
            permutation = PermuteAttrs.get_nhwc_to_nchw_permutation(input_shape_rank)

            data_node = node.in_node(2)

            name = node.soft_get('name', node.id) + '/ShapeGather'
            const = Const(graph, {'value': permutation.perm, 'name': name + '/Const',
                                  'need_shape_inference': True}).create_node_with_data()
            axis_const = Const(graph, {'value': int64_array(0), 'name': name + '/Axis'}).create_node_with_data()
            gather = Gather(graph, {'name': name,
                                    'need_shape_inference': True}).create_node_with_data([data_node, const, axis_const])
            attrs = graph.get_edge_data(data_node.id, node.id, key=0).copy()

            graph.add_edge(gather.id, node.id, **attrs)
            graph.remove_edge(data_node.id, node.id)
