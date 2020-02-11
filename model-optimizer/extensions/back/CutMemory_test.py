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

import unittest
from extensions.back.CutMemory import CutMemory
from mo.utils.unittest.graph import build_graph
from mo.utils.ir_engine.compare_graphs import compare_graphs


class CutMemoryTest(unittest.TestCase):
    def test_remove_memory(self):
        """Memory should be replaced by input and output"""
        graph = build_graph(
            nodes_attrs={
                'input': {'kind': 'op'},
                'data_in': {'kind': 'data', 'shape': None, 'value': None},
                'memory_in': {'kind': 'op', 'op': 'Memory', 'index': 1, 'id': 'memory_', 'in_ports_count': 1},
                'data_mem': {'kind': 'data', 'shape': None, 'value': None},
                'concat': {'kind': 'op', 'op': 'Concat', 'axis': 0},
                'concat_data': {'kind': 'data', 'shape': None, 'value': None},
                'some_op': {'kind': 'op'},
                'some_op_data': {'kind': 'data', 'shape': None, 'value': None},
                'memory_out': {'kind': 'op', 'op': 'Memory', 'index': 0, 'id': 'memory_'},
                'data_mem_out': {'kind': 'data', 'shape': None, 'value': None},
                'mem_out_result': {'kind': 'op', 'op': 'Result'}
            },
            edges=[
                ('input', 'data_in'), ('memory_in', 'data_mem'),
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
        CutMemory().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, last_node='mem_out_result', check_op_attrs=True)
        self.assertTrue(flag, resp)
