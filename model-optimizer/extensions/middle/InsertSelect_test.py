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
import unittest

import numpy as np

from extensions.middle.InsertSelect import AddSelectBeforeMemoryNodePattern
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph


class InsertSelectTests(unittest.TestCase):

    # graph have no splices - selects should not be inserted
    def test_insert_select_0(self):
        graph = build_graph({'in_node': {'kind': 'data', 'shape': [1, 13]},
                             'placeholder_1': {'kind': 'op', 'op': None},
                             'placeholder_data_1': {'kind': 'data', 'shape': [1, 13]},
                             'memory': {'kind': 'op', 'op': 'Memory', 'index': 0},
                             },
                            [('in_node', 'placeholder_1'), ('placeholder_1', 'placeholder_data_1'),
                             ('placeholder_data_1', 'memory')
                             ],
                            nodes_with_edges_only=True)
        AddSelectBeforeMemoryNodePattern().find_and_replace_pattern(graph)
        ref_graph = build_graph({'in_node': {'kind': 'data', 'shape': [1, 13]},
                                 'placeholder_1': {'kind': 'op', 'op': None},
                                 'placeholder_data_1': {'kind': 'data', 'shape': [1, 13]},
                                 'memory': {'kind': 'op', 'op': 'Memory', 'index': 0},
                                 },
                                [('in_node', 'placeholder_1'), ('placeholder_1', 'placeholder_data_1'),
                                 ('placeholder_data_1', 'memory')
                                 ],
                                nodes_with_edges_only=True
                                )

        (flag, resp) = compare_graphs(graph, ref_graph, 'memory')
        self.assertTrue(flag, resp)

    # graph contains 1 splice with context length 5, should be inserted select with memory as counter with length 5
    def test_insert_select_1(self):
        graph = build_graph({'in_node': {'kind': 'data', 'shape': [1, 13]},
                             'placeholder_1': {'kind': 'op', 'op': None},
                             'placeholder_data_1': {'kind': 'data', 'shape': [1, 13]},
                             'splice_1': {'kind': 'op', 'op': 'Splice', 'context': np.array([-2, -1, 0, 1, 2])},
                             'splice_data_1': {'kind': 'data', 'shape': [1, 13]},
                             'placeholder_2': {'kind': 'op', 'op': None},
                             'placeholder_data_2': {'kind': 'data', 'shape': [1, 26]},
                             'memory': {'kind': 'op', 'op': 'Memory', 'index': 0},
                             },
                            [('in_node', 'placeholder_1'), ('placeholder_1', 'placeholder_data_1'),
                             ('placeholder_data_1', 'splice_1'), ('splice_1', 'splice_data_1'),
                             ('splice_data_1', 'placeholder_2'), ('placeholder_2', 'placeholder_data_2'),
                             ('placeholder_data_2', 'memory')
                             ],
                            nodes_with_edges_only=True)
        AddSelectBeforeMemoryNodePattern().find_and_replace_pattern(graph)
        ref_graph = build_graph({'in_node': {'kind': 'data', 'shape': [1, 13]},
                                 'placeholder_1': {'kind': 'op', 'op': None},
                                 'placeholder_data_1': {'kind': 'data', 'shape': [1, 13]},
                                 'splice_1': {'kind': 'op', 'op': 'Splice', 'context': np.array([-2, -1, 0, 1, 2])},
                                 'splice_data_1': {'kind': 'data', 'shape': [1, 13]},
                                 'placeholder_2': {'kind': 'op', 'op': None},

                                 'memory_in': {'kind': 'op', 'op': 'Memory', 'shape': int64_array([5])},
                                 'memory_in_data': {'kind': 'data'},
                                 'memory_out': {'kind': 'op', 'op': 'Memory', 'shape': int64_array([5])},
                                 'memory_out_data': {'kind': 'data'},
                                 'result': {'kind': 'op', 'op': 'Result'},
                                 'crop_in': {'kind': 'op', 'op': 'Crop', 'axis': 1, 'offset': 1, 'dim': 4},
                                 'crop_in_data': {'kind': 'data'},
                                 'crop_out': {'kind': 'op', 'op': 'Crop', 'axis': 1, 'offset': 0, 'dim': 1},
                                 'crop_out_data': {'kind': 'data'},
                                 'select': {'kind': 'op', 'op': 'Select'},
                                 'select_out_data': {'kind': 'data', 'shape': [1, 26]},
                                 'const_0': {'kind': 'op', 'op': 'Const'},
                                 'const_0_data': {'kind': 'data'},
                                 'const_1': {'kind': 'op', 'op': 'Const'},
                                 'const_1_data': {'kind': 'data'},
                                 'concat': {'kind': 'op', 'op': 'Concat'},
                                 'concat_data': {'kind': 'data'},

                                 'placeholder_data_2': {'kind': 'data', 'shape': [1, 26]},
                                 'memory': {'kind': 'op', 'op': 'Memory', 'index': 0},
                                 },
                                [('in_node', 'placeholder_1'), ('placeholder_1', 'placeholder_data_1'),
                                 ('placeholder_data_1', 'splice_1'), ('splice_1', 'splice_data_1'),
                                 ('splice_data_1', 'placeholder_2'), ('placeholder_2', 'placeholder_data_2'),
                                 ('placeholder_data_2', 'select', {'in': 1}),

                                 ('memory_in', 'memory_in_data'), ('memory_in_data', 'crop_in'),
                                 ('crop_in', 'crop_in_data'), ('crop_in_data', 'concat', {'in': 0}),
                                 ('const_1', 'const_1_data'), ('const_1_data', 'concat', {'in': 1}),
                                 ('concat', 'concat_data'), ('concat_data', 'memory_out'),
                                 ('memory_out', 'memory_out_data'), ('memory_out_data', 'result'),
                                 ('concat_data', 'crop_out'), ('crop_out', 'crop_out_data'),
                                 ('crop_out_data', 'select', {'in': 0}),
                                 ('const_0', 'const_0_data'), ('const_0_data', 'select', {'in': 2}),

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
        graph = build_graph({'in_node': {'kind': 'data', 'shape': [1, 13]},
                             'placeholder_1': {'kind': 'op', 'op': None},
                             'placeholder_data_1': {'kind': 'data', 'shape': [1, 13]},
                             'splice_1': {'kind': 'op', 'op': 'Splice', 'context': np.array([-2, -1, 0, 1, 2])},
                             'splice_data_1': {'kind': 'data', 'shape': [1, 65]},
                             'splice_2': {'kind': 'op', 'op': 'Splice', 'context': np.array([-1, 0, 1])},
                             'splice_data_2': {'kind': 'data', 'shape': [1, 39]},
                             'placeholder_2': {'kind': 'op', 'op': None},
                             'placeholder_data_2': {'kind': 'data', 'shape': [1, 26]},
                             'memory': {'kind': 'op', 'op': 'Memory', 'index': 0},
                             },
                            [('in_node', 'placeholder_1'), ('placeholder_1', 'placeholder_data_1'),
                             ('placeholder_data_1', 'splice_1'), ('splice_1', 'splice_data_1'),
                             ('placeholder_data_1', 'splice_2'), ('splice_2', 'splice_data_2'),
                             ('splice_data_1', 'placeholder_2'), ('placeholder_2', 'placeholder_data_2'),
                             ('placeholder_data_2', 'memory')
                             ],
                            nodes_with_edges_only=True)
        AddSelectBeforeMemoryNodePattern().find_and_replace_pattern(graph)
        ref_graph = build_graph({'in_node': {'kind': 'data', 'shape': [1, 13]},
                                 'placeholder_1': {'kind': 'op', 'op': None},
                                 'placeholder_data_1': {'kind': 'data', 'shape': [1, 13]},
                                 'splice_1': {'kind': 'op', 'op': 'Splice', 'context': np.array([-2, -1, 0, 1, 2])},
                                 'splice_data_1': {'kind': 'data', 'shape': [1, 65]},
                                 'splice_2': {'kind': 'op', 'op': 'Splice', 'context': np.array([-1, 0, 1])},
                                 'splice_data_2': {'kind': 'data', 'shape': [1, 39]},
                                 'placeholder_2': {'kind': 'op', 'op': None},

                                 'memory_in': {'kind': 'op', 'op': 'Memory', 'shape': int64_array([5])},
                                 'memory_in_data': {'kind': 'data'},
                                 'memory_out': {'kind': 'op', 'op': 'Memory', 'shape': int64_array([5])},
                                 'memory_out_data': {'kind': 'data'},
                                 'result': {'kind': 'op', 'op': 'Result'},
                                 'crop_in': {'kind': 'op', 'op': 'Crop', 'axis': 1, 'offset': 1, 'dim': 4},
                                 'crop_in_data': {'kind': 'data'},
                                 'crop_out': {'kind': 'op', 'op': 'Crop', 'axis': 1, 'offset': 0, 'dim': 1},
                                 'crop_out_data': {'kind': 'data'},
                                 'select': {'kind': 'op', 'op': 'Select'},
                                 'select_out_data': {'kind': 'data', 'shape': [1, 26]},
                                 'const_0': {'kind': 'op', 'op': 'Const'},
                                 'const_0_data': {'kind': 'data'},
                                 'const_1': {'kind': 'op', 'op': 'Const'},
                                 'const_1_data': {'kind': 'data'},
                                 'concat': {'kind': 'op', 'op': 'Concat'},
                                 'concat_data': {'kind': 'data'},

                                 'placeholder_data_2': {'kind': 'data', 'shape': [1, 26]},
                                 'memory': {'kind': 'op', 'op': 'Memory', 'index': 0},
                                 },
                                [('in_node', 'placeholder_1'), ('placeholder_1', 'placeholder_data_1'),
                                 ('placeholder_data_1', 'splice_1'), ('splice_1', 'splice_data_1'),
                                 ('placeholder_data_1', 'splice_2'), ('splice_2', 'splice_data_2'),
                                 ('splice_data_1', 'placeholder_2'), ('placeholder_2', 'placeholder_data_2'),
                                 ('placeholder_data_2', 'select', {'in': 1}),

                                 ('memory_in', 'memory_in_data'), ('memory_in_data', 'crop_in'),
                                 ('crop_in', 'crop_in_data'), ('crop_in_data', 'concat', {'in': 0}),
                                 ('const_1', 'const_1_data'), ('const_1_data', 'concat', {'in': 1}),
                                 ('concat', 'concat_data'), ('concat_data', 'memory_out'),
                                 ('memory_out', 'memory_out_data'), ('memory_out_data', 'result'),
                                 ('concat_data', 'crop_out'), ('crop_out', 'crop_out_data'),
                                 ('crop_out_data', 'select', {'in': 0}),
                                 ('const_0', 'const_0_data'), ('const_0_data', 'select', {'in': 2}),

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
        graph = build_graph({'in_node': {'kind': 'data', 'shape': [1, 13]},
                             'placeholder_1': {'kind': 'op', 'op': None},
                             'placeholder_data_1': {'kind': 'data', 'shape': [1, 13]},
                             'splice_1': {'kind': 'op', 'op': 'Splice', 'context': np.array([-2, -1, 0, 1, 2])},
                             'splice_data_1': {'kind': 'data', 'shape': [1, 65]},
                             'splice_2': {'kind': 'op', 'op': 'Splice', 'context': np.array([-1, 0, 1])},
                             'splice_data_2': {'kind': 'data', 'shape': [1, 39]},
                             'placeholder_2': {'kind': 'op', 'op': None},
                             'placeholder_data_2': {'kind': 'data', 'shape': [1, 26]},
                             'memory': {'kind': 'op', 'op': 'Memory', 'index': 0},
                             },
                            [('in_node', 'placeholder_1'), ('placeholder_1', 'placeholder_data_1'),
                             ('placeholder_data_1', 'splice_1'), ('splice_1', 'splice_data_1'),
                             ('splice_data_1', 'splice_2'), ('splice_2', 'splice_data_2'),
                             ('splice_data_2', 'placeholder_2'), ('placeholder_2', 'placeholder_data_2'),
                             ('placeholder_data_2', 'memory')
                             ],
                            nodes_with_edges_only=True)
        AddSelectBeforeMemoryNodePattern().find_and_replace_pattern(graph)
        ref_graph = build_graph({'in_node': {'kind': 'data', 'shape': [1, 13]},
                                 'placeholder_1': {'kind': 'op', 'op': None},
                                 'placeholder_data_1': {'kind': 'data', 'shape': [1, 13]},
                                 'splice_1': {'kind': 'op', 'op': 'Splice', 'context': np.array([-2, -1, 0, 1, 2])},
                                 'splice_data_1': {'kind': 'data', 'shape': [1, 65]},
                                 'splice_2': {'kind': 'op', 'op': 'Splice', 'context': np.array([-1, 0, 1])},
                                 'splice_data_2': {'kind': 'data', 'shape': [1, 39]},
                                 'placeholder_2': {'kind': 'op', 'op': None},

                                 'memory_in': {'kind': 'op', 'op': 'Memory', 'shape': int64_array([7])},
                                 'memory_in_data': {'kind': 'data'},
                                 'memory_out': {'kind': 'op', 'op': 'Memory', 'shape': int64_array([7])},
                                 'memory_out_data': {'kind': 'data'},
                                 'result': {'kind': 'op', 'op': 'Result'},
                                 'crop_in': {'kind': 'op', 'op': 'Crop', 'axis': 1, 'offset': 1, 'dim': 6},
                                 'crop_in_data': {'kind': 'data'},
                                 'crop_out': {'kind': 'op', 'op': 'Crop', 'axis': 1, 'offset': 0, 'dim': 1},
                                 'crop_out_data': {'kind': 'data'},
                                 'select': {'kind': 'op', 'op': 'Select'},
                                 'select_out_data': {'kind': 'data', 'shape': [1, 26]},
                                 'const_0': {'kind': 'op', 'op': 'Const'},
                                 'const_0_data': {'kind': 'data'},
                                 'const_1': {'kind': 'op', 'op': 'Const'},
                                 'const_1_data': {'kind': 'data'},
                                 'concat': {'kind': 'op', 'op': 'Concat'},
                                 'concat_data': {'kind': 'data'},

                                 'placeholder_data_2': {'kind': 'data', 'shape': [1, 26]},
                                 'memory': {'kind': 'op', 'op': 'Memory', 'index': 0},
                                 },
                                [('in_node', 'placeholder_1'), ('placeholder_1', 'placeholder_data_1'),
                                 ('placeholder_data_1', 'splice_1'), ('splice_1', 'splice_data_1'),
                                 ('splice_data_1', 'splice_2'), ('splice_2', 'splice_data_2'),
                                 ('splice_data_2', 'placeholder_2'), ('placeholder_2', 'placeholder_data_2'),
                                 ('placeholder_data_2', 'select', {'in': 1}),

                                 ('memory_in', 'memory_in_data'), ('memory_in_data', 'crop_in'),
                                 ('crop_in', 'crop_in_data'), ('crop_in_data', 'concat', {'in': 0}),
                                 ('const_1', 'const_1_data'), ('const_1_data', 'concat', {'in': 1}),
                                 ('concat', 'concat_data'), ('concat_data', 'memory_out'),
                                 ('memory_out', 'memory_out_data'), ('memory_out_data', 'result'),
                                 ('concat_data', 'crop_out'), ('crop_out', 'crop_out_data'),
                                 ('crop_out_data', 'select', {'in': 0}),
                                 ('const_0', 'const_0_data'), ('const_0_data', 'select', {'in': 2}),

                                 ('select', 'select_out_data'),
                                 ('select_out_data', 'memory')
                                 ],
                                nodes_with_edges_only=True
                                )

        (flag, resp) = compare_graphs(graph, ref_graph, 'memory')
        self.assertTrue(flag, resp)
