# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.middle.InsertSelect import AddSelectBeforeMemoryNodePattern
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph


class InsertSelectTests(unittest.TestCase):

    # graph have no splices - selects should not be inserted
    def test_insert_select_0(self):
        graph = build_graph({
                             'input': {'kind': 'op', 'op': 'Parameter'},
                             'placeholder_data_1': {'kind': 'data', 'shape': [1, 13]},
                             'memory': {'kind': 'op', 'op': 'Assign'},
                             },
                            [('input', 'placeholder_data_1'),
                             ('placeholder_data_1', 'memory')
                             ],
                            nodes_with_edges_only=True)
        ref_graph = graph.copy()
        AddSelectBeforeMemoryNodePattern().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, ref_graph, 'memory')
        self.assertTrue(flag, resp)

    # graph contains 1 splice with context length 5, should be inserted select with memory as counter with length 5
    def test_insert_select_1(self):
        graph = build_graph({
                             'input': {'kind': 'op', 'op': 'Parameter'},
                             'placeholder_data_1': {'kind': 'data', 'shape': [1, 13]},
                             'splice_1': {'kind': 'op', 'op': 'Splice', 'context': np.array([-2, -1, 0, 1, 2])},
                             'splice_data_1': {'kind': 'data', 'shape': [1, 13]},
                             'placeholder_2': {'kind': 'op', 'op': None},
                             'placeholder_data_2': {'kind': 'data', 'shape': [1, 26]},
                             'memory': {'kind': 'op', 'op': 'Assign', 'index': 0},
                             },
                            [('input', 'placeholder_data_1'),
                             ('placeholder_data_1', 'splice_1'), ('splice_1', 'splice_data_1'),
                             ('splice_data_1', 'placeholder_2'), ('placeholder_2', 'placeholder_data_2'),
                             ('placeholder_data_2', 'memory')
                             ],
                            nodes_with_edges_only=True)
        AddSelectBeforeMemoryNodePattern().find_and_replace_pattern(graph)
        ref_graph = build_graph({
                                 'input': {'kind': 'op', 'op': 'Parameter'},
                                 'placeholder_data_1': {'kind': 'data', 'shape': [1, 13]},
                                 'splice_1': {'kind': 'op', 'op': 'Splice', 'context': np.array([-2, -1, 0, 1, 2])},
                                 'splice_data_1': {'kind': 'data', 'shape': [1, 13]},
                                 'placeholder_2': {'kind': 'op', 'op': None},

                                 'second_dim_mem_1': {'kind': 'op', 'op': 'Const', 'value': int64_array([5])},
                                 'second_dim_data_mem_1': {'kind': 'data'},
                                 'gather_shape_mem_1': {'kind': 'op', 'op': 'Concat'},
                                 'gather_shape_data_mem_1': {'kind': 'data'},
                                 'fill_value': {'kind': 'op', 'op': 'Const', 'value': int64_array([0])},
                                 'fill_value_data': {'kind': 'data'},
                                 'broadcast_mem_1': {'kind': 'op', 'op': 'Broadcast'},
                                 'broadcast_data_mem_1': {'kind': 'data'},

                                 'shape': {'kind': 'op', 'op': 'ShapeOf'},
                                 'shape_data': {'kind': 'data'},
                                 'crop_batch': {'kind': 'op', 'op': 'Crop', 'offset': int64_array([0])},
                                 'crop_batch_data': {'kind': 'data'},
                                 'crop_batch_dim': {'kind': 'op', 'op': 'Const', 'value': int64_array([1])},
                                 'crop_batch_dim_data': {'kind': 'data'},
                                 'second_dim': {'kind': 'op', 'op': 'Const', 'value': int64_array([5])},
                                 'second_dim_data': {'kind': 'data'},
                                 'gather_shape': {'kind': 'op', 'op': 'Concat'},
                                 'gather_shape_data': {'kind': 'data'},
                                 'fill_value_ones': {'kind': 'op', 'op': 'Const', 'value': int64_array([0])},
                                 'fill_value_data_ones': {'kind': 'data'},
                                 'broadcast': {'kind': 'op', 'op': 'Broadcast'},
                                 'broadcast_data': {'kind': 'data'},

                                 'second_dim_mem_2': {'kind': 'op', 'op': 'Const', 'value': int64_array([26])},
                                 'second_dim_data_mem_2': {'kind': 'data'},
                                 'gather_shape_mem_2': {'kind': 'op', 'op': 'Concat'},
                                 'gather_shape_data_mem_2': {'kind': 'data'},
                                 'fill_value_ones_2': {'kind': 'op', 'op': 'Const', 'value': int64_array([0])},
                                 'fill_value_data_ones_2': {'kind': 'data'},
                                 'broadcast_mem_2': {'kind': 'op', 'op': 'Broadcast'},
                                 'broadcast_data_mem_2': {'kind': 'data'},

                                 'memory_in': {'kind': 'op', 'op': 'ReadValue', 'shape': int64_array([5])},
                                 'memory_in_data': {'kind': 'data'},
                                 'memory_out': {'kind': 'op', 'op': 'Assign', 'shape': int64_array([5])},
                                 'memory_out_data': {'kind': 'data'},
                                 'result': {'kind': 'op', 'op': 'Result'},
                                 'crop_in': {'kind': 'op', 'op': 'Crop', 'axis': 1, 'offset': 1, 'dim': 4},
                                 'crop_in_data': {'kind': 'data'},
                                 'crop_out': {'kind': 'op', 'op': 'Crop', 'axis': 1, 'offset': 0, 'dim': 1},
                                 'crop_out_data': {'kind': 'data'},
                                 'equal': {'kind': 'op', 'op': 'Equal'},
                                 'equal_data': {'kind': 'data'},
                                 'select': {'kind': 'op', 'op': 'Select'},
                                 'select_out_data': {'kind': 'data', 'shape': [1, 26]},
                                 'const_0': {'kind': 'op', 'op': 'Const'},
                                 'const_0_data': {'kind': 'data'},
                                 'concat': {'kind': 'op', 'op': 'Concat'},
                                 'concat_data': {'kind': 'data'},

                                 'placeholder_data_2': {'kind': 'data', 'shape': [1, 26]},
                                 'memory': {'kind': 'op', 'op': 'Assign'},
                                 },
                                [('input', 'placeholder_data_1'),
                                 ('placeholder_data_1', 'splice_1'), ('splice_1', 'splice_data_1'),
                                 ('splice_data_1', 'placeholder_2'), ('placeholder_2', 'placeholder_data_2'),
                                ('placeholder_data_2', 'select', {'in': 1}),

                                 ('second_dim_mem_1', 'second_dim_data_mem_1'),
                                 ('second_dim_data_mem_1', 'gather_shape_mem_1', {'in': 1}),
                                 ('crop_batch_data', 'gather_shape_mem_1', {'in': 0}),
                                 ('gather_shape_mem_1', 'gather_shape_data_mem_1'),
                                 ('fill_value', 'fill_value_data'),
                                 ('fill_value_data', 'broadcast_mem_1', {'in': 0}),
                                 ('gather_shape_data_mem_1', 'broadcast_mem_1', {'in': 1}),
                                 ('broadcast_mem_1', 'broadcast_data_mem_1'),
                                 ('broadcast_data_mem_1', 'memory_in'),

                                 ('memory_in', 'memory_in_data'), ('memory_in_data', 'crop_in'),
                                 ('crop_in', 'crop_in_data'), ('crop_in_data', 'concat', {'in': 0}),

                                 ('second_dim_mem_2', 'second_dim_data_mem_2'),
                                 ('second_dim_data_mem_2', 'gather_shape_mem_2', {'in': 1}),
                                 ('crop_batch_data', 'gather_shape_mem_2', {'in': 0}),
                                 ('gather_shape_mem_2', 'gather_shape_data_mem_2'),
                                 ('fill_value_ones_2', 'fill_value_data_ones_2'),
                                 ('fill_value_data_ones_2', 'broadcast_mem_2', {'in': 0}),
                                 ('gather_shape_data_mem_2', 'broadcast_mem_2', {'in': 1}),
                                 ('broadcast_mem_2', 'broadcast_data_mem_2'),
                                 ('broadcast_data_mem_2', 'concat', {'in': 1}),

                                 ('concat', 'concat_data'), ('concat_data', 'memory_out'),
                                 ('memory_out', 'memory_out_data'), ('memory_out_data', 'result'),
                                 ('concat_data', 'crop_out'), ('crop_out', 'crop_out_data'),
                                 ('crop_out_data', 'equal', {'in': 1}), ('broadcast_data_mem_2', 'equal', {'in': 0}),
                                 ('equal', 'equal_data'),
                                 ('equal_data', 'select', {'in': 0}),

                                 ('placeholder_data_2', 'shape'), ('shape', 'shape_data'),
                                 ('shape_data', 'crop_batch'), ('crop_batch', 'crop_batch_data'),
                                 ('crop_batch_dim', 'crop_batch_dim_data'),
                                 ('crop_batch_dim_data', 'crop_batch', {'in': 1}),
                                 ('second_dim', 'second_dim_data'), ('second_dim_data', 'gather_shape', {'in': 1}),
                                 ('crop_batch_data', 'gather_shape', {'in': 0}), ('gather_shape', 'gather_shape_data'),
                                 ('fill_value_ones', 'fill_value_data_ones'),
                                 ('fill_value_data_ones', 'broadcast', {'in': 0}),
                                 ('gather_shape_data', 'broadcast', {'in': 1}), ('broadcast', 'broadcast_data'),
                                 ('broadcast_data', 'select', {'in': 2}),

                                 ('select', 'select_out_data'),
                                 ('select_out_data', 'memory')
                                 ],
                                nodes_with_edges_only=True
                                )

        (flag, resp) = compare_graphs(graph, ref_graph, 'memory')
        self.assertTrue(flag, resp)

    # graph contains 1 splice with context length 5 on the path to memory and 1 out of path,
    # should be inserted select with memory as counter with length 5
    def test_insert_select_2(self):
        graph = build_graph({
                             'input': {'kind': 'op', 'op': 'Parameter'},
                             'placeholder_data_1': {'kind': 'data', 'shape': [1, 13]},
                             'splice_1': {'kind': 'op', 'op': 'Splice', 'context': np.array([-2, -1, 0, 1, 2])},
                             'splice_data_1': {'kind': 'data', 'shape': [1, 65]},
                             'splice_2': {'kind': 'op', 'op': 'Splice', 'context': np.array([-1, 0, 1])},
                             'splice_data_2': {'kind': 'data', 'shape': [1, 39]},
                             'placeholder_2': {'kind': 'op', 'op': None},
                             'placeholder_data_2': {'kind': 'data', 'shape': [1, 26]},
                             'memory': {'kind': 'op', 'op': 'Assign'},
                             },
                            [('input', 'placeholder_data_1'),
                             ('placeholder_data_1', 'splice_1'), ('splice_1', 'splice_data_1'),
                             ('placeholder_data_1', 'splice_2'), ('splice_2', 'splice_data_2'),
                             ('splice_data_1', 'placeholder_2'), ('placeholder_2', 'placeholder_data_2'),
                             ('placeholder_data_2', 'memory')
                             ],
                            nodes_with_edges_only=True)
        AddSelectBeforeMemoryNodePattern().find_and_replace_pattern(graph)
        ref_graph = build_graph({
                                 'input': {'kind': 'op', 'op': 'Parameter'},
                                 'placeholder_data_1': {'kind': 'data', 'shape': [1, 13]},
                                 'splice_1': {'kind': 'op', 'op': 'Splice', 'context': np.array([-2, -1, 0, 1, 2])},
                                 'splice_data_1': {'kind': 'data', 'shape': [1, 65]},
                                 'splice_2': {'kind': 'op', 'op': 'Splice', 'context': np.array([-1, 0, 1])},
                                 'splice_data_2': {'kind': 'data', 'shape': [1, 39]},
                                 'placeholder_2': {'kind': 'op', 'op': None},

                                 'second_dim_mem_1': {'kind': 'op', 'op': 'Const', 'value': int64_array([5])},
                                 'second_dim_data_mem_1': {'kind': 'data'},
                                 'gather_shape_mem_1': {'kind': 'op', 'op': 'Concat'},
                                 'gather_shape_data_mem_1': {'kind': 'data'},
                                 'fill_value': {'kind': 'op', 'op': 'Const', 'value': int64_array([0])},
                                 'fill_value_data': {'kind': 'data'},
                                 'broadcast_mem_1': {'kind': 'op', 'op': 'Broadcast'},
                                 'broadcast_data_mem_1': {'kind': 'data'},

                                 'shape': {'kind': 'op', 'op': 'ShapeOf'},
                                 'shape_data': {'kind': 'data'},
                                 'crop_batch': {'kind': 'op', 'op': 'Crop', 'offset': int64_array([0])},
                                 'crop_batch_data': {'kind': 'data'},
                                 'crop_batch_dim': {'kind': 'op', 'op': 'Const', 'value': int64_array([1])},
                                 'crop_batch_dim_data': {'kind': 'data'},
                                 'second_dim': {'kind': 'op', 'op': 'Const', 'value': int64_array([5])},
                                 'second_dim_data': {'kind': 'data'},
                                 'gather_shape': {'kind': 'op', 'op': 'Concat'},
                                 'gather_shape_data': {'kind': 'data'},
                                 'fill_value_ones': {'kind': 'op', 'op': 'Const', 'value': int64_array([0])},
                                 'fill_value_data_ones': {'kind': 'data'},
                                 'broadcast': {'kind': 'op', 'op': 'Broadcast'},
                                 'broadcast_data': {'kind': 'data'},

                                 'second_dim_mem_2': {'kind': 'op', 'op': 'Const', 'value': int64_array([26])},
                                 'second_dim_data_mem_2': {'kind': 'data'},
                                 'gather_shape_mem_2': {'kind': 'op', 'op': 'Concat'},
                                 'gather_shape_data_mem_2': {'kind': 'data'},
                                 'fill_value_ones_2': {'kind': 'op', 'op': 'Const', 'value': int64_array([0])},
                                 'fill_value_data_ones_2': {'kind': 'data'},
                                 'broadcast_mem_2': {'kind': 'op', 'op': 'Broadcast'},
                                 'broadcast_data_mem_2': {'kind': 'data'},

                                 'memory_in': {'kind': 'op', 'op': 'ReadValue', 'shape': int64_array([5])},
                                 'memory_in_data': {'kind': 'data'},
                                 'memory_out': {'kind': 'op', 'op': 'Assign', 'shape': int64_array([5])},
                                 'memory_out_data': {'kind': 'data'},
                                 'result': {'kind': 'op', 'op': 'Result'},
                                 'crop_in': {'kind': 'op', 'op': 'Crop', 'axis': 1, 'offset': 1, 'dim': 4},
                                 'crop_in_data': {'kind': 'data'},
                                 'crop_out': {'kind': 'op', 'op': 'Crop', 'axis': 1, 'offset': 0, 'dim': 1},
                                 'crop_out_data': {'kind': 'data'},
                                 'equal': {'kind': 'op', 'op': 'Equal'},
                                 'equal_data': {'kind': 'data'},
                                 'select': {'kind': 'op', 'op': 'Select'},
                                 'select_out_data': {'kind': 'data', 'shape': [1, 26]},
                                 'const_0': {'kind': 'op', 'op': 'Const'},
                                 'const_0_data': {'kind': 'data'},
                                 'concat': {'kind': 'op', 'op': 'Concat'},
                                 'concat_data': {'kind': 'data'},

                                 'placeholder_data_2': {'kind': 'data', 'shape': [1, 26]},
                                 'memory': {'kind': 'op', 'op': 'Assign'},
                                 },
                                [('input', 'placeholder_data_1'),
                                 ('placeholder_data_1', 'splice_1'), ('splice_1', 'splice_data_1'),
                                 ('placeholder_data_1', 'splice_2'), ('splice_2', 'splice_data_2'),
                                 ('splice_data_1', 'placeholder_2'), ('placeholder_2', 'placeholder_data_2'),
                                 ('placeholder_data_2', 'select', {'in': 1}),

                                 ('second_dim_mem_1', 'second_dim_data_mem_1'),
                                 ('second_dim_data_mem_1', 'gather_shape_mem_1', {'in': 1}),
                                 ('crop_batch_data', 'gather_shape_mem_1', {'in': 0}),
                                 ('gather_shape_mem_1', 'gather_shape_data_mem_1'),
                                 ('fill_value', 'fill_value_data'),
                                 ('fill_value_data', 'broadcast_mem_1', {'in': 0}),
                                 ('gather_shape_data_mem_1', 'broadcast_mem_1', {'in': 1}),
                                 ('broadcast_mem_1', 'broadcast_data_mem_1'),
                                 ('broadcast_data_mem_1', 'memory_in'),

                                 ('memory_in', 'memory_in_data'), ('memory_in_data', 'crop_in'),
                                 ('crop_in', 'crop_in_data'), ('crop_in_data', 'concat', {'in': 0}),

                                 ('second_dim_mem_2', 'second_dim_data_mem_2'),
                                 ('second_dim_data_mem_2', 'gather_shape_mem_2', {'in': 1}),
                                 ('crop_batch_data', 'gather_shape_mem_2', {'in': 0}),
                                 ('gather_shape_mem_2', 'gather_shape_data_mem_2'),
                                 ('fill_value_ones_2', 'fill_value_data_ones_2'),
                                 ('fill_value_data_ones_2', 'broadcast_mem_2', {'in': 0}),
                                 ('gather_shape_data_mem_2', 'broadcast_mem_2', {'in': 1}),
                                 ('broadcast_mem_2', 'broadcast_data_mem_2'),
                                 ('broadcast_data_mem_2', 'concat', {'in': 1}),

                                 ('concat', 'concat_data'), ('concat_data', 'memory_out'),
                                 ('memory_out', 'memory_out_data'), ('memory_out_data', 'result'),
                                 ('concat_data', 'crop_out'), ('crop_out', 'crop_out_data'),
                                 ('crop_out_data', 'equal', {'in': 1}), ('broadcast_data_mem_2', 'equal', {'in': 0}),
                                 ('equal', 'equal_data'),
                                 ('equal_data', 'select', {'in': 0}),

                                 ('placeholder_data_2', 'shape'), ('shape', 'shape_data'),
                                 ('shape_data', 'crop_batch'), ('crop_batch', 'crop_batch_data'),
                                 ('crop_batch_dim', 'crop_batch_dim_data'),
                                 ('crop_batch_dim_data', 'crop_batch', {'in': 1}),
                                 ('second_dim', 'second_dim_data'), ('second_dim_data', 'gather_shape', {'in': 1}),
                                 ('crop_batch_data', 'gather_shape', {'in': 0}), ('gather_shape', 'gather_shape_data'),
                                 ('fill_value_ones', 'fill_value_data_ones'),
                                 ('fill_value_data_ones', 'broadcast', {'in': 0}),
                                 ('gather_shape_data', 'broadcast', {'in': 1}), ('broadcast', 'broadcast_data'),
                                 ('broadcast_data', 'select', {'in': 2}),

                                 ('select', 'select_out_data'),
                                 ('select_out_data', 'memory')
                                 ],
                                nodes_with_edges_only=True
                                )
        (flag, resp) = compare_graphs(graph, ref_graph, 'memory')
        self.assertTrue(flag, resp)

    # graph contains 2 splices with sum context length 8 on the path to memory,
    # should be inserted select with memory as counter with length 7
    def test_insert_select_3(self):
        graph = build_graph({
                             'input': {'kind': 'op', 'op': 'Parameter'},
                             'placeholder_data_1': {'kind': 'data', 'shape': [1, 13]},
                             'splice_1': {'kind': 'op', 'op': 'Splice', 'context': np.array([-2, -1, 0, 1, 2])},
                             'splice_data_1': {'kind': 'data', 'shape': [1, 65]},
                             'splice_2': {'kind': 'op', 'op': 'Splice', 'context': np.array([-1, 0, 1])},
                             'splice_data_2': {'kind': 'data', 'shape': [1, 39]},
                             'placeholder_2': {'kind': 'op', 'op': None},
                             'placeholder_data_2': {'kind': 'data', 'shape': [1, 26]},
                             'memory': {'kind': 'op', 'op': 'Assign', 'index': 0},
                             },
                            [('input', 'placeholder_data_1'),
                             ('placeholder_data_1', 'splice_1'), ('splice_1', 'splice_data_1'),
                             ('splice_data_1', 'splice_2'), ('splice_2', 'splice_data_2'),
                             ('splice_data_2', 'placeholder_2'), ('placeholder_2', 'placeholder_data_2'),
                             ('placeholder_data_2', 'memory')
                             ],
                            nodes_with_edges_only=True)
        AddSelectBeforeMemoryNodePattern().find_and_replace_pattern(graph)
        ref_graph = build_graph({
                                 'input': {'kind': 'op', 'op': 'Parameter'},
                                 'placeholder_data_1': {'kind': 'data', 'shape': [1, 13]},
                                 'splice_1': {'kind': 'op', 'op': 'Splice', 'context': np.array([-2, -1, 0, 1, 2])},
                                 'splice_data_1': {'kind': 'data', 'shape': [1, 65]},
                                 'splice_2': {'kind': 'op', 'op': 'Splice', 'context': np.array([-1, 0, 1])},
                                 'splice_data_2': {'kind': 'data', 'shape': [1, 39]},
                                 'placeholder_2': {'kind': 'op', 'op': None},

                                 'second_dim_mem_1': {'kind': 'op', 'op': 'Const', 'value': int64_array([5])},
                                 'second_dim_data_mem_1': {'kind': 'data'},
                                 'gather_shape_mem_1': {'kind': 'op', 'op': 'Concat'},
                                 'gather_shape_data_mem_1': {'kind': 'data'},
                                 'fill_value': {'kind': 'op', 'op': 'Const', 'value': int64_array([0])},
                                 'fill_value_data': {'kind': 'data'},
                                 'broadcast_mem_1': {'kind': 'op', 'op': 'Broadcast'},
                                 'broadcast_data_mem_1': {'kind': 'data'},

                                 'shape': {'kind': 'op', 'op': 'ShapeOf'},
                                 'shape_data': {'kind': 'data'},
                                 'crop_batch': {'kind': 'op', 'op': 'Crop', 'offset': int64_array([0])},
                                 'crop_batch_data': {'kind': 'data'},
                                 'crop_batch_dim': {'kind': 'op', 'op': 'Const', 'value': int64_array([1])},
                                 'crop_batch_dim_data': {'kind': 'data'},
                                 'second_dim': {'kind': 'op', 'op': 'Const', 'value': int64_array([5])},
                                 'second_dim_data': {'kind': 'data'},
                                 'gather_shape': {'kind': 'op', 'op': 'Concat'},
                                 'gather_shape_data': {'kind': 'data'},
                                 'fill_value_ones': {'kind': 'op', 'op': 'Const', 'value': int64_array([0])},
                                 'fill_value_data_ones': {'kind': 'data'},
                                 'broadcast': {'kind': 'op', 'op': 'Broadcast'},
                                 'broadcast_data': {'kind': 'data'},

                                 'second_dim_mem_2': {'kind': 'op', 'op': 'Const', 'value': int64_array([26])},
                                 'second_dim_data_mem_2': {'kind': 'data'},
                                 'gather_shape_mem_2': {'kind': 'op', 'op': 'Concat'},
                                 'gather_shape_data_mem_2': {'kind': 'data'},
                                 'fill_value_ones_2': {'kind': 'op', 'op': 'Const', 'value': int64_array([0])},
                                 'fill_value_data_ones_2': {'kind': 'data'},
                                 'broadcast_mem_2': {'kind': 'op', 'op': 'Broadcast'},
                                 'broadcast_data_mem_2': {'kind': 'data'},

                                 'memory_in': {'kind': 'op', 'op': 'ReadValue', 'shape': int64_array([5])},
                                 'memory_in_data': {'kind': 'data'},
                                 'memory_out': {'kind': 'op', 'op': 'Assign', 'shape': int64_array([5])},
                                 'memory_out_data': {'kind': 'data'},
                                 'result': {'kind': 'op', 'op': 'Result'},
                                 'crop_in': {'kind': 'op', 'op': 'Crop', 'axis': 1, 'offset': 1, 'dim': 4},
                                 'crop_in_data': {'kind': 'data'},
                                 'crop_out': {'kind': 'op', 'op': 'Crop', 'axis': 1, 'offset': 0, 'dim': 1},
                                 'crop_out_data': {'kind': 'data'},
                                 'equal': {'kind': 'op', 'op': 'Equal'},
                                 'equal_data': {'kind': 'data'},
                                 'select': {'kind': 'op', 'op': 'Select'},
                                 'select_out_data': {'kind': 'data', 'shape': [1, 26]},
                                 'const_0': {'kind': 'op', 'op': 'Const'},
                                 'const_0_data': {'kind': 'data'},
                                 'concat': {'kind': 'op', 'op': 'Concat'},
                                 'concat_data': {'kind': 'data'},

                                 'placeholder_data_2': {'kind': 'data', 'shape': [1, 26]},
                                 'memory': {'kind': 'op', 'op': 'Assign', 'index': 0},
                                 },
                                [('input', 'placeholder_data_1'),
                                 ('placeholder_data_1', 'splice_1'), ('splice_1', 'splice_data_1'),
                                 ('splice_data_1', 'splice_2'), ('splice_2', 'splice_data_2'),
                                 ('splice_data_2', 'placeholder_2'), ('placeholder_2', 'placeholder_data_2'),
                                 ('placeholder_data_2', 'select', {'in': 1}),

                                 ('second_dim_mem_1', 'second_dim_data_mem_1'),
                                 ('second_dim_data_mem_1', 'gather_shape_mem_1', {'in': 1}),
                                 ('crop_batch_data', 'gather_shape_mem_1', {'in': 0}),
                                 ('gather_shape_mem_1', 'gather_shape_data_mem_1'),
                                 ('fill_value', 'fill_value_data'),
                                 ('fill_value_data', 'broadcast_mem_1', {'in': 0}),
                                 ('gather_shape_data_mem_1', 'broadcast_mem_1', {'in': 1}),
                                 ('broadcast_mem_1', 'broadcast_data_mem_1'),
                                 ('broadcast_data_mem_1', 'memory_in'),

                                 ('memory_in', 'memory_in_data'), ('memory_in_data', 'crop_in'),
                                 ('crop_in', 'crop_in_data'), ('crop_in_data', 'concat', {'in': 0}),

                                 ('second_dim_mem_2', 'second_dim_data_mem_2'),
                                 ('second_dim_data_mem_2', 'gather_shape_mem_2', {'in': 1}),
                                 ('crop_batch_data', 'gather_shape_mem_2', {'in': 0}),
                                 ('gather_shape_mem_2', 'gather_shape_data_mem_2'),
                                 ('fill_value_ones_2', 'fill_value_data_ones_2'),
                                 ('fill_value_data_ones_2', 'broadcast_mem_2', {'in': 0}),
                                 ('gather_shape_data_mem_2', 'broadcast_mem_2', {'in': 1}),
                                 ('broadcast_mem_2', 'broadcast_data_mem_2'),
                                 ('broadcast_data_mem_2', 'concat', {'in': 1}),

                                 ('concat', 'concat_data'), ('concat_data', 'memory_out'),
                                 ('memory_out', 'memory_out_data'), ('memory_out_data', 'result'),
                                 ('concat_data', 'crop_out'), ('crop_out', 'crop_out_data'),
                                 ('crop_out_data', 'equal', {'in': 1}), ('broadcast_data_mem_2', 'equal', {'in': 0}),
                                 ('equal', 'equal_data'),
                                 ('equal_data', 'select', {'in': 0}),

                                 ('placeholder_data_2', 'shape'), ('shape', 'shape_data'),
                                 ('shape_data', 'crop_batch'), ('crop_batch', 'crop_batch_data'),
                                 ('crop_batch_dim', 'crop_batch_dim_data'),
                                 ('crop_batch_dim_data', 'crop_batch', {'in': 1}),
                                 ('second_dim', 'second_dim_data'), ('second_dim_data', 'gather_shape', {'in': 1}),
                                 ('crop_batch_data', 'gather_shape', {'in': 0}), ('gather_shape', 'gather_shape_data'),
                                 ('fill_value_ones', 'fill_value_data_ones'),
                                 ('fill_value_data_ones', 'broadcast', {'in': 0}),
                                 ('gather_shape_data', 'broadcast', {'in': 1}), ('broadcast', 'broadcast_data'),
                                 ('broadcast_data', 'select', {'in': 2}),

                                 ('select', 'select_out_data'),
                                 ('select_out_data', 'memory')
                                 ],
                                nodes_with_edges_only=True
                                )

        (flag, resp) = compare_graphs(graph, ref_graph, 'memory')
        self.assertTrue(flag, resp)
