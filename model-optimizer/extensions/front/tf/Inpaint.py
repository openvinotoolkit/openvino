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
import numpy as np

from extensions.middle.InsertLayoutPropagationTransposes import mark_as_correct_data_layout
from extensions.ops.normalize_l2 import NormalizeL2Op
from extensions.ops.transpose import Transpose
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.front.tf.replacement import FrontReplacementFromConfigFileGeneral
from mo.graph.graph import Graph, Node, rename_nodes
from mo.middle.pattern_match import apply_pattern
from mo.ops.const import Const
from mo.ops.reshape import Reshape

INPAINT_EIP1_PATTERN = {
    'nodes': [
        ('eip', dict(op='ExtractImagePatches')),
        ('reshape', dict(op='Reshape')),
        ('transpose', dict(op='Transpose')),
        ('split', dict(op='Split')),
        ('ss', dict(op='StridedSlice')),
        ('square', dict(op='Pow')),
        ('sum', dict(op='ReduceSum')),
        ('sqrt', dict(op='Pow')),
        ('max', dict(op='Maximum')),
        ('div', dict(op='Div')),
        ('reshape_const', dict(op='Const')),
        ('transpose_const', dict(op='Const')),
        ('sum_const', dict(op='Const')),
    ],
    'edges': [
        ('eip', 'reshape', {'out': 0, 'in': 0}),
        ('reshape_const', 'reshape', {'out': 0, 'in': 1}),
        ('reshape', 'transpose', {'out': 0, 'in': 0}),
        ('transpose_const', 'transpose', {'out': 0, 'in': 1}),
        ('transpose', 'split', {'out': 0, 'in': 0}),
        ('split', 'ss', {'out': 0, 'in': 0}),
        ('ss', 'square', {'out': 0, 'in': 0}),
        ('ss', 'div', {'out': 0, 'in': 0}),
        ('square', 'sum', {'out': 0, 'in': 0}),
        ('sum_const', 'sum', {'out': 0, 'in': 1}),
        ('sum', 'sqrt', {'out': 0, 'in': 0}),
        ('sqrt', 'max', {'out': 0, 'in': 0}),
        ('max', 'div', {'out': 0, 'in': 1}),
    ]
}


INPAINT_EIP2_PATTERN = {
    'nodes': [
        ('eip', dict(op='ExtractImagePatches')),
        ('reshape', dict(op='Reshape')),
        ('transpose', dict(op='Transpose')),
        ('ss', dict(op='StridedSlice')),
        ('mean', dict(op='ReduceMean')),
        ('reshape_const', dict(op='Const')),
        ('transpose_const', dict(op='Const')),
        ('mean_const', dict(op='Const')),
    ],
    'edges': [
        ('eip', 'reshape', {'out': 0, 'in': 0}),
        ('reshape_const', 'reshape', {'out': 0, 'in': 1}),
        ('reshape', 'transpose', {'out': 0, 'in': 0}),
        ('transpose_const', 'transpose', {'out': 0, 'in': 1}),
        ('transpose', 'ss', {'out': 0, 'in': 0}),
        ('ss', 'mean', {'out': 0, 'in': 0}),
        ('mean_const', 'mean', {'out': 0, 'in': 1}),
    ]
}


class InpaintTransformation(FrontReplacementFromConfigFileGeneral):
    replacement_id = 'Inpaint'

    def transform_graph(self, graph: Graph, replacement_descriptions):
        apply_pattern(graph, **INPAINT_EIP1_PATTERN, action=self.optimize_eip1)
        apply_pattern(graph, **INPAINT_EIP2_PATTERN, action=self.optimize_eip2)

    @staticmethod
    def optimize_eip1(graph: Graph, match: dict):
        eip = match['eip']
        div = match['div']
        maximum = match['max']

        eip.out_port(0).disconnect()
        new_reshape = create_op_node_with_second_input(graph, Reshape, int64_array([-1, 3 * 3 * 96]), input_node=eip)
        normalize_l2 = create_op_node_with_second_input(graph, NormalizeL2Op, int64_array([1]),
                                                        {'eps_mode': 'max', 
                                                         'eps': maximum.in_port(1).get_source().node.value})
        normalize_l2.in_port(0).connect(new_reshape.out_port(0))
        final_reshape = create_op_node_with_second_input(graph, Reshape, int64_array([-1, 3, 3, 96]), input_node=normalize_l2)
        mark_as_correct_data_layout(final_reshape)
        transpose = create_op_node_with_second_input(graph, Transpose, int64_array([1, 2, 3, 0]), input_node=final_reshape)
        mark_as_correct_data_layout(transpose)
        div.out_port(0).get_connection().set_source(transpose.out_port(0))

        rename_nodes([(div, div.id + '/TBR'), (final_reshape, div.soft_get('name', div.id))])

    @staticmethod
    def optimize_eip2(graph: Graph, match: dict):
        reshape_const = match['reshape_const']
        transpose_const = match['transpose_const']
        mean_const = match['mean_const']
        if not np.array_equal(reshape_const.value, [1, -1, 3, 3, 1]):
            return
        if not np.array_equal(transpose_const.value, [0, 2, 3, 4, 1]):
            return
        if not np.array_equal(mean_const.value, [0, 1, 2]):
            return

        mean = match['mean']
        eip = match['eip']

        mean.in_port(0).disconnect()
        eip.out_port(0).get_connection().set_destination(mean.in_port(0))

        mean.in_port(1).disconnect()
        mean.in_port(1).connect(Const(graph, {'value': int64_array([-1])}).create_node().out_port(0))
        mean.out_port(0).get_connection().insert_node(
            create_op_node_with_second_input(graph, Reshape, int64_array([1, 1, 1, -1])))

        graph.remove_nodes_from([match['transpose'].id, match['ss'].id])
