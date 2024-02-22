# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.back.CutMemory import CutMemoryInput, CutMemoryOutput
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph


class CutMemoryTest(unittest.TestCase):
    def test_remove_memory(self):
        """Memory should be replaced by input and output"""
        graph = build_graph(
            nodes_attrs={
                'input': {'kind': 'op'},
                'data_in': {'kind': 'data', 'shape': None, 'value': None},
                'const_0': {'kind': 'op', 'op': 'Const'},
                'const_0_data': {'kind': 'data'},
                'memory_in': {'kind': 'op', 'op': 'ReadValue', 'variable_id': 'memory_'},
                'data_mem': {'kind': 'data', 'shape': None, 'value': None},
                'concat': {'kind': 'op', 'op': 'Concat', 'axis': 0},
                'concat_data': {'kind': 'data', 'shape': None, 'value': None},
                'some_op': {'kind': 'op'},
                'some_op_data': {'kind': 'data', 'shape': None, 'value': None},
                'memory_out': {'kind': 'op', 'op': 'Assign', 'variable_id': 'memory_'},
                'data_mem_out': {'kind': 'data', 'shape': None, 'value': None},
                'mem_out_result': {'kind': 'op', 'op': 'Result'}
            },
            edges=[
                ('input', 'data_in'),
                ('const_0', 'const_0_data'), ('const_0_data', 'memory_in'), ('memory_in', 'data_mem'),
                ('data_in', 'concat', {'in': 0}), ('data_mem', 'concat', {'in': 1}),
                ('concat', 'concat_data'), ('concat_data', 'some_op'),
                ('some_op', 'some_op_data'), ('some_op_data', 'memory_out'),
                ('memory_out', 'data_mem_out'), ('data_mem_out', 'mem_out_result')
            ]
        )
        graph_ref = build_graph(
            nodes_attrs={
                'input': {'kind': 'op'},
                'data_in': {'kind': 'data', 'shape': None, 'value': None},
                'new_input': {'kind': 'op', 'op': 'Parameter'},
                'new_in_data': {'kind': 'data', 'shape': None, 'value': None},
                'concat': {'kind': 'op', 'op': 'Concat', 'axis': 0},
                'concat_data': {'kind': 'data', 'shape': None, 'value': None},
                'some_op': {'kind': 'op'},
                'some_op_data': {'kind': 'data', 'shape': None, 'value': None},
                'crop': {'kind': 'op', 'op': 'Crop', 'axis': np.array([0])},
                'crop_data': {'kind': 'data', 'shape': None, 'value': None},
                'mem_out_result': {'kind': 'op', 'op': 'Result'},
            },
            edges=[
                ('input', 'data_in'), ('new_input', 'new_in_data'),
                ('data_in', 'concat', {'in': 0}), ('new_in_data', 'concat', {'in': 1}),
                ('concat', 'concat_data'), ('concat_data', 'some_op'),
                ('some_op', 'some_op_data'), ('some_op_data', 'crop'),
                ('crop', 'crop_data'), ('crop_data', 'mem_out_result')
            ],
        )
        CutMemoryInput().find_and_replace_pattern(graph)
        CutMemoryOutput().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, last_node='mem_out_result', check_op_attrs=True)
        self.assertTrue(flag, resp)
