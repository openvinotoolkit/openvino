"""
 Copyright (c) 2018-2019 Intel Corporation

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
from extensions.back.InsertLayoutPropagationTransposes import mark_as_correct_data_layout, \
    mark_input_as_in_correct_layout, mark_output_as_in_correct_layout
from extensions.ops.transpose import Transpose
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
from mo.ops.reshape import Reshape


class DepthToSpace(MiddleReplacementPattern):
    """
    Replaces DepthToSpace with 6D_Reshape->Transpose->4D_Reshape sequence
    """

    enabled = True

    def run_after(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def run_before(self):
        from extensions.middle.pass_separator import MiddleFinish
        return [MiddleFinish]

    def pattern(self):
        return dict(
            nodes=[
                ('in_data', dict(kind='data')),
                ('op', dict(op='DepthToSpace', data_format='NHWC')),
                ('out_data', dict(kind='data'))
            ],
            edges=[
                ('in_data', 'op'),
                ('op', 'out_data')
            ])

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['op']

        N, H, W, C = match['in_data'].shape
        block_size = node['block_size']

        graph.remove_edge(match['in_data'].id, node.id)
        graph.remove_edge(node.id, match['out_data'].id)

        dim_6D = int64_array([N, block_size, block_size, int(C / (block_size ** 2)), H, W])
        order_6D = int64_array([0, 3, 4, 1, 5, 2])
        dim_4D = int64_array([N, int(H * block_size), int(W * block_size), int(C / (block_size ** 2))])

        reshape_6_op = Reshape(graph, dict(name=node.id + '/Reshape_to_6D'))
        reshape_6_const_data = Const(graph, dict(value=dim_6D)).create_node_with_data()
        reshape_6_data_node = reshape_6_op.create_node_with_data([match['in_data'], reshape_6_const_data])
        mark_as_correct_data_layout(reshape_6_data_node.in_node(0))

        order_const_data = Const(graph, dict(value=order_6D)).create_node_with_data()

        transpose_op = Transpose(graph, dict(name=node.id + '/Transpose'))
        transpose_data_node = transpose_op.create_node_with_data([reshape_6_data_node, order_const_data])
        mark_as_correct_data_layout(transpose_data_node.in_node(0))

        reshape_4_op = Reshape(graph, dict(name=node.id + '/Reshape_to_4D'))
        reshape_4_const_data = Const(graph, dict(value=dim_4D)).create_node_with_data()
        reshape_4_data_node = reshape_4_op.create_node_with_data([transpose_data_node, reshape_4_const_data],
                                                                 data_nodes=[match['out_data']])
        mark_input_as_in_correct_layout(reshape_4_data_node.in_node(0), 0)
        mark_output_as_in_correct_layout(reshape_4_data_node.in_node(0), 0)

