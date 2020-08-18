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
from mo.front.tf.graph_utils import create_op_node_with_second_input, create_op_with_const_inputs
from mo.front.tf.replacement import FrontReplacementFromConfigFileGeneral
from mo.graph.graph import Graph, Node, rename_nodes
from mo.middle.pattern_match import apply_pattern
from mo.ops.const import Const
from mo.ops.reshape import Reshape
from mo.ops.strided_slice import StridedSlice

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

INPAINT_EIP3_PATTERN = {
    'nodes': [
        ('eip', dict(op='ExtractImagePatches')),
        ('reshape', dict(op='Reshape')),
        ('transpose', dict(op='Transpose')),
        ('split', dict(op='Split')),
        ('ss', dict(op='StridedSlice')),
        ('reshape_const', dict(op='Const')),
        ('transpose_const', dict(op='Const')),
    ],
    'edges': [
        ('eip', 'reshape', {'out': 0, 'in': 0}),
        ('reshape_const', 'reshape', {'out': 0, 'in': 1}),
        ('reshape', 'transpose', {'out': 0, 'in': 0}),
        ('transpose_const', 'transpose', {'out': 0, 'in': 1}),
        ('transpose', 'split', {'out': 0, 'in': 0}),
        ('split', 'ss', {'out': 0, 'in': 0}),
    ]
}


class InpaintTransformation(FrontReplacementFromConfigFileGeneral):
    replacement_id = 'Inpaint'

    def transform_graph(self, graph: Graph, replacement_descriptions):
        apply_pattern(graph, **INPAINT_EIP1_PATTERN, action=self.optimize_eip1)
        apply_pattern(graph, **INPAINT_EIP2_PATTERN, action=self.optimize_eip2)
        apply_pattern(graph, **INPAINT_EIP3_PATTERN, action=self.optimize_eip3)

    @staticmethod
    def optimize_eip1(graph: Graph, match: dict):
        eip = match['eip']
        div = match['div']
        maximum = match['max']
        r_const = match['reshape_const']

        div_name = div.soft_get('name', div.id)
        eip.out_port(0).disconnect()
        new_ss = create_op_with_const_inputs(graph, StridedSlice,
                                             {1: int64_array([1]), 2: int64_array([5]), 3: int64_array([1])},
                                             {'begin_mask': int64_array([1]), 'end_mask': int64_array([1]),
                                              'new_axis_mask': int64_array([0]), 'shrink_axis_mask': int64_array([0]),
                                              'ellipsis_mask': int64_array([0])},
                                             input_node=r_const)
        reshape = Reshape(graph, {'name': div_name + '/Reshape'}).create_node()
        reshape.in_port(0).connect(eip.out_port(0))
        reshape.in_port(1).connect(new_ss.out_port(0))
        mark_as_correct_data_layout(reshape)
        norm_reshape = create_op_node_with_second_input(graph, Reshape, int64_array([0, -1]),
                                                        {'name': div_name + '/Norm/Reshape'}, input_node=reshape)
        mark_as_correct_data_layout(norm_reshape)
        normalize_l2 = create_op_node_with_second_input(graph, NormalizeL2Op, int64_array([1]),
                                                        {'name': div_name + '/Norm', 'eps_mode': 'max',
                                                         'eps': maximum.in_port(1).get_source().node.value})
        normalize_l2.in_port(0).connect(norm_reshape.out_port(0))
        final_reshape = Reshape(graph, {'name': div_name + '/Reshape'}).create_node()
        final_reshape.in_port(0).connect(normalize_l2.out_port(0))
        final_reshape.in_port(1).connect(new_ss.out_port(0))
        mark_as_correct_data_layout(final_reshape)
        transpose = create_op_node_with_second_input(graph, Transpose, int64_array([1, 2, 3, 0]),
                                                     {'name': div_name + '/Transpose'}, input_node=final_reshape)
        mark_as_correct_data_layout(transpose)
        div.out_port(0).get_connection().set_source(transpose.out_port(0))

        rename_nodes([(div, div_name + '/TBR'), (transpose, div_name)])

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

    @staticmethod
    def optimize_eip3(graph: Graph, match: dict):
        eip = match['eip']
        ss = match['ss']
        r_const = match['reshape_const']
        ss_name = ss.soft_get('name', ss.id)

        eip.out_port(0).disconnect()
        new_ss = create_op_with_const_inputs(graph, StridedSlice,
                                             {1: int64_array([1]), 2: int64_array([5]), 3: int64_array([1])},
                                             {'begin_mask': int64_array([1]), 'end_mask': int64_array([1]),
                                              'new_axis_mask': int64_array([0]), 'shrink_axis_mask': int64_array([0]),
                                              'ellipsis_mask': int64_array([0])},
                                             input_node=r_const)
        final_reshape = Reshape(graph, {}).create_node()
        final_reshape.in_port(0).connect(eip.out_port(0))
        final_reshape.in_port(1).connect(new_ss.out_port(0))

        mark_as_correct_data_layout(final_reshape)
        transpose = create_op_node_with_second_input(graph, Transpose, int64_array([0, 3, 1, 2]),
                                                     input_node=final_reshape)
        mark_as_correct_data_layout(transpose)
        ss.out_port(0).get_connection().set_source(transpose.out_port(0))

        rename_nodes([(ss, ss_name + '/TBR'), (transpose, ss_name)])
