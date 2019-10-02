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
import unittest

from extensions.middle.ReplaceSpliceNodePattern import ReplaceSpliceNodePattern
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph, compare_graphs


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
                                 'memory_in': {'kind': 'op', 'op': 'Memory'},
                                 'memory_in_data': {'kind': 'data'},
                                 'crop_mem':  {'kind': 'op', 'op': 'Crop', 'offset': 13, 'dim': 130},
                                 'crop_mem_data': {'kind': 'data'},
                                 'concat': {'kind': 'op', 'op': 'Concat'},
                                 'concat_data': {'kind': 'data', 'shape': [1, 143]},
                                 'memory_out': {'kind': 'op', 'op': 'Memory'},
                                 'memory_out_data': {'kind': 'data'},
                                 'result': {'kind': 'op', 'op': 'Result'},
                                 'out_placeholder': {'kind': 'op', 'op': 'placeholder'},
                                 },
                                [
                                    ('in_placeholder', 'in_node'),
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
                                 'memory_in': {'kind': 'op', 'op': 'Memory'},
                                 'memory_in_data': {'kind': 'data'},
                                 'crop_mem': {'kind': 'op', 'op': 'Crop', 'offset': 3, 'dim': 30},
                                 'crop_mem_data': {'kind': 'data'},
                                 'concat': {'kind': 'op', 'op': 'Concat'},
                                 'concat_data': {'kind': 'data'},
                                 'memory_out': {'kind': 'op', 'op': 'Memory'},
                                 'memory_out_data': {'kind': 'data'},
                                 'result': {'kind': 'op', 'op': 'Result'},
                                 'memory_in_constdims': {'kind': 'op', 'op': 'Memory'},
                                 'memory_in_constdims_data': {'kind': 'data'},
                                 'crop_mem_constdims': {'kind': 'op', 'op': 'Crop', 'offset': 10, 'dim': 100},
                                 'crop_mem_constdims_data': {'kind': 'data'},
                                 'concat_constdims': {'kind': 'op', 'op': 'Concat'},
                                 'concat_constdims_data': {'kind': 'data'},
                                 'memory_out_constdims': {'kind': 'op', 'op': 'Memory'},
                                 'memory_out_constdims_data': {'kind': 'data'},
                                 'result_constdims': {'kind': 'op', 'op': 'Result'},
                                 'crop_first_constdims': {'kind': 'op', 'op': 'Crop', 'offset': 0, 'dim': 10},
                                 'crop_first_constdims_data': {'kind': 'data'},
                                 'concat_all': {'kind': 'op', 'op': 'Concat'},
                                 'concat_all_data': {'kind': 'data', 'shape': [1, 43]},
                                 'out_placeholder': {'kind': 'op', 'op': 'placeholder'},
                                 },
                                [
                                    ('in_placeholder', 'in_node'),
                                    ('in_node', 'split'),
                                    ('split', 'split_data_0', {'out': 0}),
                                    ('split', 'split_data_1', {'out': 1}),
                                    ('memory_in', 'memory_in_data'),
                                    ('memory_in_data', 'crop_mem'),
                                    ('crop_mem', 'crop_mem_data'),
                                    ('crop_mem_data', 'concat', {'in': 0}),
                                    ('split_data_0', 'concat', {'in': 1}),
                                    ('concat', 'concat_data'),
                                    ('concat_data', 'memory_out'),
                                    ('memory_out', 'memory_out_data'),
                                    ('memory_out_data', 'result'),
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
                                ]
                                )

        (flag, resp) = compare_graphs(graph, ref_graph, 'out_placeholder')
        self.assertTrue(flag, resp)
