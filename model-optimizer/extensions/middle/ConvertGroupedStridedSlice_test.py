"""
 Copyright (c) 2018 Intel Corporation

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

from extensions.middle.ConvertGroupedStridedSlice import ConvertGroupedStridedSlice
from mo.utils.unittest.graph import build_graph, compare_graphs

nodes_attributes = {
    'placeholder_1': {'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # StridedSlice layers
    'sslice_1': {'type': None, 'kind': 'op', 'op': 'StridedSlice', 'slices': None,
                 'shrink_axis_mask': np.array([False, False, False, False])},
    'sslice_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'sslice_2': {'type': None, 'kind': 'op', 'op': 'StridedSlice', 'slices': None,
                 'shrink_axis_mask': np.array([False, False, False, False])},
    'sslice_2_data': {'value': None, 'shape': None, 'kind': 'data'},
    'sslice_3': {'type': None, 'kind': 'op', 'op': 'StridedSlice', 'slices': None,
                 'shrink_axis_mask': np.array([False, False, False, False])},
    'sslice_3_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Split layer
    'split_1': {'type': 'Split', 'kind': 'op', 'op': 'SplitV'},
    'split_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'split_2_data': {'value': None, 'shape': None, 'kind': 'data'},
    'split_3_data': {'value': None, 'shape': None, 'kind': 'data'},
    'split_4_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Concat1 operation
    'concat_1': {'type': 'Concat', 'kind': 'op', 'op': 'Concat'},
    'concat_1_data': {'value': None, 'shape': None, 'kind': 'data'},
}


class ConvertGroupedStridedSliceTests(unittest.TestCase):
    def test_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_1'),
                             ('sslice_1', 'sslice_1_data'),
                             ('placeholder_1_data', 'sslice_2'),
                             ('sslice_2', 'sslice_2_data'),
                             ('placeholder_1_data', 'sslice_3'),
                             ('sslice_3', 'sslice_3_data'),
                             ('sslice_1_data', 'concat_1'),
                             ('sslice_2_data', 'concat_1'),
                             ('sslice_3_data', 'concat_1'),
                             ('concat_1', 'concat_1_data')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},

                             'sslice_1': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(0, 18, 1)])},
                             'sslice_1_data': {'shape': np.array([1, 227, 227, 18])},

                             'sslice_2': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(18, 36, 1)])},
                             'sslice_2_data': {'shape': np.array([1, 227, 227, 18])},

                             'sslice_3': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(36, 54, 1)])},
                             'sslice_3_data': {'shape': np.array([1, 227, 227, 18])},

                             'concat_1_data': {'shape': np.array([1, 227, 227, 54]), 'is_output': True},
                             })
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'split_1'),
                                 ('split_1', 'split_1_data'),
                                 ('split_1', 'split_2_data'),
                                 ('split_1', 'split_3_data'),
                                 ('split_1_data', 'concat_1'),
                                 ('split_2_data', 'concat_1'),
                                 ('split_3_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},
                                 'split_1': {'axis': 3},
                                 'split_1_data': {'shape': np.array([1, 227, 227, 18])},
                                 'split_2_data': {'shape': np.array([1, 227, 227, 18])},
                                 'split_3_data': {'shape': np.array([1, 227, 227, 18])},
                                 'concat_1_data': {'shape': np.array([1, 227, 227, 54]), 'is_output': True},
                                 })

        pattern = ConvertGroupedStridedSlice()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_2(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_1'),
                             ('sslice_1', 'sslice_1_data'),
                             ('placeholder_1_data', 'sslice_2'),
                             ('sslice_2', 'sslice_2_data'),
                             ('placeholder_1_data', 'sslice_3'),
                             ('sslice_3', 'sslice_3_data'),
                             ('sslice_1_data', 'concat_1'),
                             ('sslice_2_data', 'concat_1'),
                             ('sslice_3_data', 'concat_1'),
                             ('concat_1', 'concat_1_data')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},

                             'sslice_1': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(19, 37, 1)])},
                             'sslice_1_data': {'shape': np.array([1, 227, 227, 18])},

                             'sslice_2': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(37, 54, 1)])},
                             'sslice_2_data': {'shape': np.array([1, 227, 227, 17])},

                             'sslice_3': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(0, 19, 1)])},
                             'sslice_3_data': {'shape': np.array([1, 227, 227, 19])},

                             'concat_1_data': {'shape': np.array([1, 227, 227, 54]), 'is_output': True},
                             })
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'split_1'),
                                 ('split_1', 'split_1_data'),
                                 ('split_1', 'split_2_data'),
                                 ('split_1', 'split_3_data'),
                                 ('split_1_data', 'concat_1'),
                                 ('split_2_data', 'concat_1'),
                                 ('split_3_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},
                                 'split_1': {'axis': 3},
                                 'split_1_data': {'shape': np.array([1, 227, 227, 18])},
                                 'split_2_data': {'shape': np.array([1, 227, 227, 17])},
                                 'split_3_data': {'shape': np.array([1, 227, 227, 19])},
                                 'concat_1_data': {'shape': np.array([1, 227, 227, 54]), 'is_output': True},
                                 })

        pattern = ConvertGroupedStridedSlice()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    # Intersection of split ranges in feature dimension
    def test_3_neg(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_1'),
                             ('sslice_1', 'sslice_1_data'),
                             ('placeholder_1_data', 'sslice_2'),
                             ('sslice_2', 'sslice_2_data'),
                             ('placeholder_1_data', 'sslice_3'),
                             ('sslice_3', 'sslice_3_data'),
                             ('sslice_1_data', 'concat_1'),
                             ('sslice_2_data', 'concat_1'),
                             ('sslice_3_data', 'concat_1'),
                             ('concat_1', 'concat_1_data')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},

                             'sslice_1': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(19, 39, 1)])},
                             'sslice_1_data': {'shape': np.array([1, 227, 227, 20])},

                             'sslice_2': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(37, 54, 1)])},
                             'sslice_2_data': {'shape': np.array([1, 227, 227, 17])},

                             'sslice_3': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(0, 19, 1)])},
                             'sslice_3_data': {'shape': np.array([1, 227, 227, 19])},

                             'concat_1_data': {'shape': np.array([1, 227, 227, 54]), 'is_output': True},
                             })
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'sslice_1'),
                                 ('sslice_1', 'sslice_1_data'),
                                 ('placeholder_1_data', 'sslice_2'),
                                 ('sslice_2', 'sslice_2_data'),
                                 ('placeholder_1_data', 'sslice_3'),
                                 ('sslice_3', 'sslice_3_data'),
                                 ('sslice_1_data', 'concat_1'),
                                 ('sslice_2_data', 'concat_1'),
                                 ('sslice_3_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},

                                 'sslice_1': {'slices': np.array(
                                     [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(19, 39, 1)])},
                                 'sslice_1_data': {'shape': np.array([1, 227, 227, 20])},

                                 'sslice_2': {'slices': np.array(
                                     [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(37, 54, 1)])},
                                 'sslice_2_data': {'shape': np.array([1, 227, 227, 17])},

                                 'sslice_3': {'slices': np.array(
                                     [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(0, 19, 1)])},
                                 'sslice_3_data': {'shape': np.array([1, 227, 227, 19])},

                                 'concat_1_data': {'shape': np.array([1, 227, 227, 54]), 'is_output': True},
                                 })

        pattern = ConvertGroupedStridedSlice()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    # Split range overflow in feature dimension
    def test_4_neg(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_1'),
                             ('sslice_1', 'sslice_1_data'),
                             ('placeholder_1_data', 'sslice_2'),
                             ('sslice_2', 'sslice_2_data'),
                             ('placeholder_1_data', 'sslice_3'),
                             ('sslice_3', 'sslice_3_data'),
                             ('sslice_1_data', 'concat_1'),
                             ('sslice_2_data', 'concat_1'),
                             ('sslice_3_data', 'concat_1'),
                             ('concat_1', 'concat_1_data')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},

                             'sslice_1': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(19, 37, 1)])},
                             'sslice_1_data': {'shape': np.array([1, 227, 227, 18])},

                             'sslice_2': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(37, 55, 1)])},
                             'sslice_2_data': {'shape': np.array([1, 227, 227, 18])},

                             'sslice_3': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(0, 19, 1)])},
                             'sslice_3_data': {'shape': np.array([1, 227, 227, 19])},

                             'concat_1_data': {'shape': np.array([1, 227, 227, 54]), 'is_output': True},
                             })
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'sslice_1'),
                                 ('sslice_1', 'sslice_1_data'),
                                 ('placeholder_1_data', 'sslice_2'),
                                 ('sslice_2', 'sslice_2_data'),
                                 ('placeholder_1_data', 'sslice_3'),
                                 ('sslice_3', 'sslice_3_data'),
                                 ('sslice_1_data', 'concat_1'),
                                 ('sslice_2_data', 'concat_1'),
                                 ('sslice_3_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},

                                 'sslice_1': {'slices': np.array(
                                     [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(19, 37, 1)])},
                                 'sslice_1_data': {'shape': np.array([1, 227, 227, 18])},

                                 'sslice_2': {'slices': np.array(
                                     [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(37, 55, 1)])},
                                 'sslice_2_data': {'shape': np.array([1, 227, 227, 18])},

                                 'sslice_3': {'slices': np.array(
                                     [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(0, 19, 1)])},
                                 'sslice_3_data': {'shape': np.array([1, 227, 227, 19])},

                                 'concat_1_data': {'shape': np.array([1, 227, 227, 54]), 'is_output': True},
                                 })

        pattern = ConvertGroupedStridedSlice()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    # Split(1,H,W,54)--->Fake_data (1,H,W,1)
    #       |`---->Sslice1_out (1,H,W,18)
    #       |`---->Sslice2_out (1,H,W,18)
    #       `----->Sslice3_out (1,H,W,17)
    def test_5(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_1'),
                             ('sslice_1', 'sslice_1_data'),
                             ('placeholder_1_data', 'sslice_2'),
                             ('sslice_2', 'sslice_2_data'),
                             ('placeholder_1_data', 'sslice_3'),
                             ('sslice_3', 'sslice_3_data'),
                             ('sslice_1_data', 'concat_1'),
                             ('sslice_2_data', 'concat_1'),
                             ('sslice_3_data', 'concat_1'),
                             ('concat_1', 'concat_1_data')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},

                             'sslice_1': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(19, 37, 1)])},
                             'sslice_1_data': {'shape': np.array([1, 227, 227, 18])},

                             'sslice_2': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(37, 54, 1)])},
                             'sslice_2_data': {'shape': np.array([1, 227, 227, 17])},

                             'sslice_3': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(1, 19, 1)])},
                             'sslice_3_data': {'shape': np.array([1, 227, 227, 18])},

                             'concat_1_data': {'shape': np.array([1, 227, 227, 54]), 'is_output': True},
                             })
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'split_1'),
                                 ('split_1', 'split_1_data'),
                                 ('split_1', 'split_2_data'),
                                 ('split_1', 'split_3_data'),
                                 ('split_1', 'split_4_data'),
                                 ('split_2_data', 'concat_1'),
                                 ('split_3_data', 'concat_1'),
                                 ('split_4_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},
                                 'split_1': {'axis': 3},
                                 'split_1_data': {'shape': np.array([1, 227, 227, 1])},
                                 'split_2_data': {'shape': np.array([1, 227, 227, 18])},
                                 'split_3_data': {'shape': np.array([1, 227, 227, 17])},
                                 'split_4_data': {'shape': np.array([1, 227, 227, 18])},
                                 'concat_1_data': {'shape': np.array([1, 227, 227, 54]), 'is_output': True},
                                 })

        pattern = ConvertGroupedStridedSlice()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    # Split(1,H,W,54)
    #       |`---->Sslice1_out (1,H,W,(0,18))
    #       |`---->Fake_data (1,H,W,(18,27))
    #       |`---->Sslice3_out (1,H,W,(27,45))
    #       `----->Fake_data (1,H,W,(45,54))
    def test_6(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_1'),
                             ('sslice_1', 'sslice_1_data'),
                             ('placeholder_1_data', 'sslice_2'),
                             ('sslice_2', 'sslice_2_data'),
                             ('sslice_1_data', 'concat_1'),
                             ('sslice_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},

                             'sslice_1': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(0, 18, 1)])},
                             'sslice_1_data': {'shape': np.array([1, 227, 227, 18])},

                             'sslice_2': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(27, 45, 1)])},
                             'sslice_2_data': {'shape': np.array([1, 227, 227, 18])},

                             'concat_1_data': {'shape': np.array([1, 227, 227, 54]), 'is_output': True},
                             })
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'split_1'),
                                 ('split_1', 'split_1_data'),
                                 ('split_1', 'split_2_data'),
                                 ('split_1', 'split_3_data'),
                                 ('split_1', 'split_4_data'),
                                 ('split_1_data', 'concat_1'),
                                 ('split_3_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},
                                 'split_1': {'axis': 3},
                                 'split_1_data': {'shape': np.array([1, 227, 227, 18])},
                                 'split_2_data': {'shape': np.array([1, 227, 227, 9])},
                                 'split_3_data': {'shape': np.array([1, 227, 227, 18])},
                                 'split_4_data': {'shape': np.array([1, 227, 227, 9])},
                                 'concat_1_data': {'shape': np.array([1, 227, 227, 54]), 'is_output': True},
                                 })

        pattern = ConvertGroupedStridedSlice()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_7_neg(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_1'),
                             ('sslice_1', 'sslice_1_data'),
                             ('placeholder_1_data', 'sslice_2'),
                             ('sslice_2', 'sslice_2_data'),
                             ('sslice_1_data', 'concat_1'),
                             ('sslice_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},

                             'sslice_1': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 10, 1), slice(0, 227, 1), slice(0, 18, 1)])},
                             'sslice_1_data': {'shape': np.array([1, 10, 227, 18])},

                             'sslice_2': {'slices': np.array(
                                 [slice(0, 1, 1), slice(10, 227, 1), slice(0, 227, 1), slice(27, 45, 1)])},
                             'sslice_2_data': {'shape': np.array([1, 217, 227, 18])},

                             'concat_1_data': {'shape': np.array([1, 227, 227, 54]), 'is_output': True},
                             })
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'sslice_1'),
                                 ('sslice_1', 'sslice_1_data'),
                                 ('placeholder_1_data', 'sslice_2'),
                                 ('sslice_2', 'sslice_2_data'),
                                 ('sslice_1_data', 'concat_1'),
                                 ('sslice_2_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},

                                 'sslice_1': {'slices': np.array(
                                     [slice(0, 1, 1), slice(0, 10, 1), slice(0, 227, 1), slice(0, 18, 1)])},
                                 'sslice_1_data': {'shape': np.array([1, 10, 227, 18])},

                                 'sslice_2': {'slices': np.array(
                                     [slice(0, 1, 1), slice(10, 227, 1), slice(0, 227, 1), slice(27, 45, 1)])},
                                 'sslice_2_data': {'shape': np.array([1, 217, 227, 18])},

                                 'concat_1_data': {'shape': np.array([1, 227, 227, 54]), 'is_output': True},
                                 })

        pattern = ConvertGroupedStridedSlice()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    # Split(1,54,W,C)
    #       |`---->Sslice1_out (1,(0,18),W,C)
    #       |`---->Sslice2_out (1,(18,36),W,C)
    #       `----->Fake_data (1,(36,54),W,C)
    def test_8(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_1'),
                             ('sslice_1', 'sslice_1_data'),
                             ('placeholder_1_data', 'sslice_2'),
                             ('sslice_2', 'sslice_2_data'),
                             ('sslice_1_data', 'concat_1'),
                             ('sslice_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 54, 54, 3])},

                             'sslice_1': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 18, 1), slice(0, 54, 1), slice(0, 3, 1)])},
                             'sslice_1_data': {'shape': np.array([1, 18, 54, 3])},

                             'sslice_2': {'slices': np.array(
                                 [slice(0, 1, 1), slice(18, 36, 1), slice(0, 54, 1), slice(0, 3, 1)])},
                             'sslice_2_data': {'shape': np.array([1, 18, 54, 3])},

                             'concat_1_data': {'shape': np.array([1, 54, 54, 3]), 'is_output': True},
                             })
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'split_1'),
                                 ('split_1', 'split_1_data'),
                                 ('split_1', 'split_2_data'),
                                 ('split_1', 'split_3_data'),
                                 ('split_1_data', 'concat_1'),
                                 ('split_3_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 54, 54, 3])},
                                 'split_1': {'axis': 1},
                                 'split_1_data': {'shape': np.array([1, 18, 54, 3])},
                                 'split_2_data': {'shape': np.array([1, 18, 54, 3])},
                                 'split_3_data': {'shape': np.array([1, 18, 54, 3])},
                                 'concat_1_data': {'shape': np.array([1, 54, 54, 3]), 'is_output': True},
                                 })

        pattern = ConvertGroupedStridedSlice()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)


if __name__ == '__main__':
    unittest.main()
