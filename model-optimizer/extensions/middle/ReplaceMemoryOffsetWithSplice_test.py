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

from extensions.middle.ReplaceMemoryOffsetWithSplice import ReplaceMemoryOffsetNodePattern
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph
from mo.utils.ir_engine.compare_graphs import compare_graphs


class ReplaceMemoryOffsetNodePatternTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes_attributes = {
            'in_placeholder': {'kind': 'op', 'op': 'placeholder'},
            'in_node': {'kind': 'data', 'shape': [1, 13]},
            'memoryoffset': {'kind': 'op', 'op': 'MemoryOffset', 't': -5,
                             'pair_name': 'memoryoffset_2', 'has_default': False},
            'memoryoffset_data': {'kind': 'data', 'shape': [1, 13]},
            'memoryoffset_2': {'kind': 'op', 'op': 'MemoryOffset', 't': -5,
                               'pair_name': 'memoryoffset', 'has_default': False,
                               'in_ports_count': 1},
            'memoryoffset_2_data': {'kind': 'data', 'shape': [1, 13]},
            'crop_data': {'kind': 'data', 'shape': [1, 13]},
            'out_placeholder': {'kind': 'op', 'op': 'placeholder'},
            'opoutput': {'kind': 'op', 'op': 'OpOutput'},
        }

    def test_memoryoffset_pos(self):
        graph = build_graph(self.nodes_attributes,
                            [('in_placeholder', 'in_node'),
                             ('in_node', 'memoryoffset'),
                             ('memoryoffset', 'memoryoffset_data'),
                             ('memoryoffset_data', 'opoutput'),
                             ('memoryoffset_2', 'memoryoffset_2_data'),
                             ('memoryoffset_2_data', 'out_placeholder')])
        memoryoffset_node = Node(graph, 'memoryoffset')
        memoryoffset_node['t'] = 5
        ReplaceMemoryOffsetNodePattern().find_and_replace_pattern(graph)
        ref_graph = build_graph({'in_placeholder': {'kind': 'op', 'op': 'placeholder'},
                                 'in_node': {'kind': 'data', 'shape': [1, 13]},
                                 'splice': {'kind': 'op', 'op': 'Splice', 'context': range(0, 6)},
                                 'splice_data': {'kind': 'data', 'shape': [1, 78]},
                                 'crop': {'kind': 'op', 'op': 'Crop', 'offset': 130, 'dim': 13},
                                 'crop_data': {'kind': 'data', 'shape': [1, 13]},
                                 'out_placeholder': {'kind': 'op', 'op': 'placeholder'},
                                 },
                                [
                                    ('in_placeholder', 'in_node'),
                                    ('in_node', 'splice'),
                                    ('splice', 'splice_data'),
                                    ('splice_data', 'crop'),
                                    ('crop', 'crop_data'),
                                    ('crop_data', 'out_placeholder')
                                ]
                                )

        (flag, resp) = compare_graphs(graph, ref_graph, 'out_placeholder')
        self.assertTrue(flag, resp)

    def test_memoryoffset_neg(self):
        graph = build_graph(self.nodes_attributes,
                            [('in_placeholder', 'in_node'),
                             ('in_node', 'memoryoffset'),
                             ('memoryoffset', 'memoryoffset_data'),
                             ('memoryoffset_data', 'opoutput'),
                             ('memoryoffset_2', 'memoryoffset_2_data'),
                             ('memoryoffset_2_data', 'out_placeholder')])
        memoryoffset_node = Node(graph, 'memoryoffset')
        memoryoffset_node['t'] = -5
        ReplaceMemoryOffsetNodePattern().find_and_replace_pattern(graph)
        ref_graph = build_graph({'in_placeholder': {'kind': 'op', 'op': 'placeholder'},
                                 'in_node': {'kind': 'data', 'shape': [1, 13]},
                                 'splice': {'kind': 'op', 'op': 'Splice', 'context': range(-5, 1)},
                                 'splice_data': {'kind': 'data', 'shape': [1, 78]},
                                 'crop': {'kind': 'op', 'op': 'Crop', 'offset': 0, 'dim': 13},
                                 'memoryoffset_2_data': {'kind': 'data', 'shape': [1, 13]},
                                 'out_placeholder': {'kind': 'op', 'op': 'placeholder'},
                                 },
                                [
                                    ('in_placeholder', 'in_node'),
                                    ('in_node', 'splice'),
                                    ('splice', 'splice_data'),
                                    ('splice_data', 'crop'),
                                    ('crop', 'memoryoffset_2_data'),
                                    ('memoryoffset_2_data', 'out_placeholder')
                                ]
                                )
        (flag, resp) = compare_graphs(graph, ref_graph, 'out_placeholder')
        self.assertTrue(flag, resp)

    def test_memoryoffset_neg_0(self):
        graph = build_graph(self.nodes_attributes,
                            [('in_placeholder', 'in_node'),
                             ('in_node', 'memoryoffset'),
                             ('memoryoffset', 'memoryoffset_data'),
                             ('memoryoffset_data', 'opoutput'),
                             ('memoryoffset_2', 'memoryoffset_2_data'),
                             ('memoryoffset_2_data', 'out_placeholder'),
                             ('in_node', 'out_placeholder')])
        memoryoffset_node = Node(graph, 'memoryoffset')
        memoryoffset_node['t'] = -5
        ReplaceMemoryOffsetNodePattern().find_and_replace_pattern(graph)
        ref_graph = build_graph({'in_placeholder': {'kind': 'op', 'op': 'placeholder'},
                                 'in_node': {'kind': 'data', 'shape': [1, 13]},
                                 'splice': {'kind': 'op', 'op': 'Splice', 'context': range(-5, 1)},
                                 'splice_data': {'kind': 'data', 'shape': [1, 78]},
                                 'crop': {'kind': 'op', 'op': 'Crop', 'offset': 0, 'dim': 13},
                                 'crop_input': {'kind': 'op', 'op': 'Crop', 'offset': 65, 'dim': 13},
                                 'crop_input_data': {'kind': 'data', 'shape': [1, 13]},
                                 'memoryoffset_2_data': {'kind': 'data', 'shape': [1, 13]},
                                 'out_placeholder': {'kind': 'op', 'op': 'placeholder'},
                                 },
                                [
                                    ('in_placeholder', 'in_node'),
                                    ('in_node', 'splice'),
                                    ('splice', 'splice_data'),
                                    ('splice_data', 'crop'),
                                    ('crop', 'memoryoffset_2_data'),
                                    ('splice_data', 'crop_input'),
                                    ('crop_input', 'crop_input_data'),
                                    ('memoryoffset_2_data', 'out_placeholder'),
                                    ('crop_input_data', 'out_placeholder')
                                ]
                                )

        (flag, resp) = compare_graphs(graph, ref_graph, 'out_placeholder')
        self.assertTrue(flag, resp)
