# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.middle.ReplaceSpliceNodePattern import ReplaceSpliceNodePattern
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph


class ReplaceSpliceNodePatternTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes_attributes = {
            'placeholder': {'kind': 'op', 'op': None},
            'in_node': {'kind': 'data', 'shape': [1, 13]},
            'splice': {'kind': 'op', 'op': 'Splice', 'context': range(-5, 6), 'const_dim': 0},
            'splice_data': {'kind': 'data', 'shape': [1, 143]},
            'out_placeholder': {'kind': 'op', 'op': 'placeholder'},
        }

    def test_splice(self):
        graph = build_graph(self.nodes_attributes,
                            [('placeholder', 'in_node'),
                             ('in_node', 'splice'),
                             ('splice', 'splice_data'),
                             ('splice_data', 'out_placeholder')])
        ReplaceSpliceNodePattern().find_and_replace_pattern(graph)

        ref_graph = build_graph({'in_placeholder': {'kind': 'op', 'op': None},
                                 'in_node': {'kind': 'data', 'shape': [1, 13]},

                                 'fill_value': {'kind': 'op', 'op': 'Const', 'value': int64_array([0])},
                                 'fill_value_data': {'kind': 'data'},

                                 'memory_in': {'kind': 'op', 'op': 'ReadValue'},
                                 'memory_in_data': {'kind': 'data'},
                                 'crop_mem':  {'kind': 'op', 'op': 'Crop', 'offset': 13, 'dim': 130},
                                 'crop_mem_data': {'kind': 'data'},
                                 'concat': {'kind': 'op', 'op': 'Concat'},
                                 'concat_data': {'kind': 'data', 'shape': [1, 143]},
                                 'memory_out': {'kind': 'op', 'op': 'Assign'},
                                 'memory_out_data': {'kind': 'data'},
                                 'result': {'kind': 'op', 'op': 'Result'},
                                 'out_placeholder': {'kind': 'op', 'op': 'placeholder'},
                                 },
                                [
                                    ('in_placeholder', 'in_node'),

                                    ('fill_value', 'fill_value_data'), ('fill_value_data', 'memory_in'),

                                    ('memory_in', 'memory_in_data'),
                                    ('memory_in_data', 'crop_mem'),
                                    ('crop_mem', 'crop_mem_data'),
                                    ('crop_mem_data', 'concat', {'in': 0}),
                                    ('in_node', 'concat', {'in': 1}),
                                    ('concat', 'concat_data'),
                                    ('concat_data', 'memory_out'),
                                    ('memory_out', 'memory_out_data'),
                                    ('memory_out_data', 'result'),
                                    ('concat_data', 'out_placeholder'),
                                ]
                                )

        (flag, resp) = compare_graphs(graph, ref_graph, 'out_placeholder')
        self.assertTrue(flag, resp)

    def test_splice_with_constdim(self):
        graph = build_graph(self.nodes_attributes,
                            [('placeholder', 'in_node'),
                             ('in_node', 'splice'),
                             ('splice', 'splice_data'),
                             ('splice_data', 'out_placeholder')])
        Node(graph, 'splice')['const_dim'] = 10
        Node(graph, 'splice_data')['shape'] = [1, 43]
        ReplaceSpliceNodePattern().find_and_replace_pattern(graph)

        ref_graph = build_graph({'in_placeholder': {'kind': 'op', 'op': None},
                                 'in_node': {'kind': 'data', 'shape': [1, 13]},
                                 'split': {'kind': 'op', 'op': 'Split'},
                                 'split_data_0': {'kind': 'data'},
                                 'split_data_1': {'kind': 'data'},

                                 'fill_value': {'kind': 'op', 'op': 'Const', 'value': int64_array([0])},
                                 'fill_value_data': {'kind': 'data'},

                                 'memory_in': {'kind': 'op', 'op': 'ReadValue'},
                                 'memory_in_data': {'kind': 'data'},
                                 'crop_mem': {'kind': 'op', 'op': 'Crop', 'offset': 3, 'dim': 30},
                                 'crop_mem_data': {'kind': 'data'},
                                 'concat': {'kind': 'op', 'op': 'Concat'},
                                 'concat_data': {'kind': 'data'},
                                 'memory_out': {'kind': 'op', 'op': 'Assign'},
                                 'memory_out_data': {'kind': 'data'},
                                 'result': {'kind': 'op', 'op': 'Result'},


                                 'fill_value_2': {'kind': 'op', 'op': 'Const', 'value': int64_array([0])},
                                 'fill_value_2_data': {'kind': 'data'},
\
                                 'memory_in_constdims': {'kind': 'op', 'op': 'ReadValue'},
                                 'memory_in_constdims_data': {'kind': 'data'},
                                 'crop_mem_constdims': {'kind': 'op', 'op': 'Crop', 'offset': 10, 'dim': 100},
                                 'crop_mem_constdims_data': {'kind': 'data'},
                                 'concat_constdims': {'kind': 'op', 'op': 'Concat'},
                                 'concat_constdims_data': {'kind': 'data'},
                                 'memory_out_constdims': {'kind': 'op', 'op': 'Assign'},
                                 'memory_out_constdims_data': {'kind': 'data'},
                                 'result_constdims': {'kind': 'op', 'op': 'Result'},
                                 'crop_first_constdims': {'kind': 'op', 'op': 'Crop', 'offset': 0, 'dim': 10},
                                 'crop_first_constdims_data': {'kind': 'data'},
                                 'concat_all': {'kind': 'op', 'op': 'Concat'},
                                 'concat_all_data': {'kind': 'data', 'shape': [1, 43]},
                                 'out_placeholder': {'kind': 'op', 'op': 'placeholder'},

                                 'axis_const': {'kind': 'op'},
                                 'axis_const_data': {'value': None, 'shape': None, 'kind': 'data'},
                                 'split_dim_const': {'kind': 'op'},
                                 'split_dim_const_data': {'value': None, 'shape': None, 'kind': 'data'},

                                 },
                                [
                                    ('in_placeholder', 'in_node'),
                                    ('in_node', 'split', {'in': 0}),
                                    ('split', 'split_data_0', {'out': 0}),
                                    ('split', 'split_data_1', {'out': 1}),

                                    ('fill_value', 'fill_value_data'), ('fill_value_data', 'memory_in'),

                                    ('memory_in', 'memory_in_data'),
                                    ('memory_in_data', 'crop_mem'),
                                    ('crop_mem', 'crop_mem_data'),
                                    ('crop_mem_data', 'concat', {'in': 0}),
                                    ('split_data_0', 'concat', {'in': 1}),
                                    ('concat', 'concat_data'),
                                    ('concat_data', 'memory_out'),
                                    ('memory_out', 'memory_out_data'),
                                    ('memory_out_data', 'result'),

                                    ('fill_value_2', 'fill_value_2_data'), ('fill_value_2_data', 'memory_in_constdims'),

                                    ('memory_in_constdims', 'memory_in_constdims_data'),
                                    ('memory_in_constdims_data', 'crop_mem_constdims'),
                                    ('crop_mem_constdims', 'crop_mem_constdims_data'),
                                    ('crop_mem_constdims_data', 'concat_constdims', {'in': 0}),
                                    ('split_data_1', 'concat_constdims', {'in': 1}),
                                    ('concat_constdims', 'concat_constdims_data'),
                                    ('concat_constdims_data', 'memory_out_constdims'),
                                    ('memory_out_constdims', 'memory_out_constdims_data'),
                                    ('memory_out_constdims_data', 'result_constdims'),
                                    ('concat_constdims_data', 'crop_first_constdims'),
                                    ('crop_first_constdims', 'crop_first_constdims_data'),
                                    ('crop_first_constdims_data', 'concat_all', {'in': 1}),
                                    ('concat_data', 'concat_all', {'in': 0}),
                                    ('concat_all', 'concat_all_data'),
                                    ('concat_all_data', 'out_placeholder'),

                                    ('axis_const', 'axis_const_data'),
                                    ('split_dim_const', 'split_dim_const_data'),
                                    ('axis_const_data', 'split', {'in': 1}),
                                    ('split_dim_const_data', 'split', {'in': 2}),

                                ]
                                )

        (flag, resp) = compare_graphs(graph, ref_graph, 'out_placeholder')
        self.assertTrue(flag, resp)
