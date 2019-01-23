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

import numpy as np
import unittest

from extensions.middle.AddReshapeAfterStridedSlice import AddReshapeAfterStridedSlice
from mo.graph.graph import Node
from mo.middle.passes.fusing.fuse_linear_ops_test import compare_graphs
from mo.middle.passes.eliminate_test import build_graph

# The dictionary with nodes attributes used to build various graphs. A key is the name of the node and the value is the
# dictionary with node attributes.
nodes_attributes_test = {
    'placeholder_1': {'shape': None, 'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
    'placeholder_1_data': {'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_2': {'shape': None, 'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
    'placeholder_2_data': {'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_begin_data': {'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_end_data': {'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_stride_data': {'shape': None, 'kind': 'data', 'data_type': None},
    # StridedSlice layers
    'sslice_1': {'type': 'StridedSlice', 'kind': 'op', 'op': 'StridedSlice', 'slices': None,
                 'shrink_axis_mask': np.array([False, False, True, False]),
                 'new_axis_mask': np.array([False, False, False, False])},
    'sslice_1_data': {'shape': None, 'kind': 'data'},
    'sslice_2': {'type': 'StridedSlice', 'kind': 'op', 'op': 'StridedSlice', 'slices': None,
                 'shrink_axis_mask': np.array([False, False, True, False]),
                 'new_axis_mask': np.array([False, False, False, False])},
    'sslice_2_data': {'shape': None, 'kind': 'data'}}

nodes_reshape = {
    'placeholder_1': {'shape': None, 'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
    'placeholder_1_data': {'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_2': {'shape': None, 'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
    'placeholder_2_data': {'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_begin_data': {'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_end_data': {'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_stride_data': {'shape': None, 'kind': 'data', 'data_type': None},
    # StridedSlice layers
    'sslice_1': {'type': 'StridedSlice', 'value': None, 'kind': 'op', 'op': 'StridedSlice', 'slices': None,
                 'shrink_axis_mask': np.array([False, False, True, False]),
                 'new_axis_mask': np.array([False, False, False, False])},
    'sslice_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'sslice_2': {'type': 'StridedSlice', 'value': None, 'kind': 'op', 'op': 'StridedSlice', 'slices': None,
                 'shrink_axis_mask': np.array([False, False, True, False]),
                 'new_axis_mask': np.array([False, False, False, False])},
    'sslice_2_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Reshape layer
    'sslice_1/Reshape_shrink': {'type': 'Reshape', 'value': None, 'kind': 'op', 'op': 'Reshape'},
    'sslice_1/Reshape_shrink_data': {'value': None, 'shape': None, 'kind': 'data'},
    'sslice_2/Reshape_shrink': {'type': 'Reshape', 'value': None, 'kind': 'op', 'op': 'Reshape'},
    'sslice_2/Reshape_shrink_data': {'value': None, 'shape': None, 'kind': 'data'},
    'sslice_2/Reshape_new': {'type': 'Reshape', 'value': None, 'kind': 'op', 'op': 'Reshape'},
    'sslice_2/Reshape_new_data': {'value': None, 'shape': None, 'kind': 'data'},
}


class AddReshapeAfterStridedSliceTests(unittest.TestCase):
    def test_ss_1_shrink_last(self):
        graph = build_graph(nodes_attributes_test,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_1'),
                             ('placeholder_begin_data', 'sslice_1'),
                             ('placeholder_end_data', 'sslice_1'),
                             ('placeholder_stride_data', 'sslice_1'),
                             ('sslice_1', 'sslice_1_data')],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},
                             'sslice_1': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 1, 1), slice(0, 54, 1)])},
                             'sslice_1_data': {'shape': np.array([1, 227, 54]), 'is_output': True},
                             })
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_reshape,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'sslice_1'),
                                 ('placeholder_begin_data', 'sslice_1'),
                                 ('placeholder_end_data', 'sslice_1'),
                                 ('placeholder_stride_data', 'sslice_1'),
                                 ('sslice_1', 'sslice_1/Reshape_shrink_data'),
                                 ('sslice_1/Reshape_shrink_data', 'sslice_1/Reshape_shrink'),
                                 ('sslice_1/Reshape_shrink', 'sslice_1_data')],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},
                                 'sslice_1': {'slices': np.array(
                                     [slice(0, 1, 1), slice(0, 227, 1), slice(0, 1, 1), slice(0, 54, 1)]),
                                     'shrink_axis_mask': np.array([False, False, False, False]),
                                     'new_axis_mask': np.array([False, False, False, False])},
                                 'sslice_1_data': {'shape': np.array([1, 227, 54]), 'is_output': True},
                                 'sslice_1/Reshape_shrink': {'dim': np.array([1, 227, 54])},
                                 'sslice_1/Reshape_shrink_data': {'shape': np.array([1, 227, 1, 54])}
                                 })

        pattern = AddReshapeAfterStridedSlice()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'sslice_1_data', check_op_attrs=True)
        graph.clear()
        graph_ref.clear()
        self.assertTrue(flag, resp)

    def test_ss_1_shrink(self):
        graph = build_graph(nodes_attributes_test,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_2'),
                             ('placeholder_begin_data', 'sslice_2'),
                             ('placeholder_end_data', 'sslice_2'),
                             ('placeholder_stride_data', 'sslice_2'),
                             ('sslice_2', 'sslice_2_data'),
                             ('sslice_2_data', 'placeholder_2'),
                             ('placeholder_2', 'placeholder_2_data'), ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},
                             'sslice_2': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 1, 1), slice(0, 54, 1)]), },
                             'sslice_2_data': {'shape': np.array([1, 227, 54]), 'is_output': True}
                             })
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_reshape,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'sslice_2'),
                                 ('placeholder_begin_data', 'sslice_2'),
                                 ('placeholder_end_data', 'sslice_2'),
                                 ('placeholder_stride_data', 'sslice_2'),
                                 ('sslice_2', 'sslice_2/Reshape_shrink_data'),
                                 ('sslice_2/Reshape_shrink_data', 'sslice_2/Reshape_shrink'),
                                 ('sslice_2/Reshape_shrink', 'sslice_2_data'),
                                 ('sslice_2_data', 'placeholder_2'),
                                 ('placeholder_2', 'placeholder_2_data')],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},
                                 'sslice_2': {'slices': np.array(
                                     [slice(0, 1, 1), slice(0, 227, 1), slice(0, 1, 1), slice(0, 54, 1)]),
                                     'shrink_axis_mask': np.array([False, False, False, False]),
                                     'new_axis_mask': np.array([False, False, False, False])},
                                 'sslice_2_data': {'shape': np.array([1, 227, 54])},
                                 'sslice_2/Reshape_shrink': {'dim': np.array([1, 227, 54])},
                                 'sslice_2/Reshape_shrink_data': {'shape': np.array([1, 227, 1, 54])},
                                 })

        pattern = AddReshapeAfterStridedSlice()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'sslice_2_data', check_op_attrs=True)
        graph.clear()
        graph_ref.clear()
        self.assertTrue(flag, resp)

    def test_ss_2_shrink(self):
        graph = build_graph(nodes_attributes_test,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_2'),
                             ('placeholder_begin_data', 'sslice_2'),
                             ('placeholder_end_data', 'sslice_2'),
                             ('placeholder_stride_data', 'sslice_2'),
                             ('sslice_2', 'sslice_2_data'),
                             ('sslice_2_data', 'placeholder_2'),
                             ('placeholder_2', 'placeholder_2_data'), ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},
                             'sslice_2': {
                                 'slices': np.array([slice(0, 1, 1), slice(0, 1, 1), slice(0, 227, 1), slice(0, 1, 1)]),
                                 'shrink_axis_mask': np.array([False, True, False, True])},
                             'sslice_2_data': {'shape': np.array([1, 227]), 'is_output': True}
                             })
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_reshape,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'sslice_2'),
                                 ('placeholder_begin_data', 'sslice_2'),
                                 ('placeholder_end_data', 'sslice_2'),
                                 ('placeholder_stride_data', 'sslice_2'),
                                 ('sslice_2', 'sslice_2/Reshape_shrink_data'),
                                 ('sslice_2/Reshape_shrink_data', 'sslice_2/Reshape_shrink'),
                                 ('sslice_2/Reshape_shrink', 'sslice_2_data'),
                                 ('sslice_2_data', 'placeholder_2'),
                                 ('placeholder_2', 'placeholder_2_data')],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},
                                 'sslice_2': {'slices': np.array(
                                     [slice(0, 1, 1), slice(0, 1, 1), slice(0, 227, 1), slice(0, 1, 1)]),
                                     'shrink_axis_mask': np.array([False, False, False, False]),
                                     'new_axis_mask': np.array([False, False, False, False])},
                                 'sslice_2_data': {'shape': np.array([1, 227])},
                                 'sslice_2/Reshape_shrink': {'dim': np.array([1, 227])},
                                 'sslice_2/Reshape_shrink_data': {'shape': np.array([1, 1, 227, 1])},
                                 })

        pattern = AddReshapeAfterStridedSlice()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'sslice_2_data', check_op_attrs=True)
        graph.clear()
        graph_ref.clear()
        self.assertTrue(flag, resp)

    def test_ss_1_new(self):
        graph = build_graph(nodes_attributes_test,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_2'),
                             ('placeholder_begin_data', 'sslice_2'),
                             ('placeholder_end_data', 'sslice_2'),
                             ('placeholder_stride_data', 'sslice_2'),
                             ('sslice_2', 'sslice_2_data'),
                             ('sslice_2_data', 'placeholder_2'),
                             ('placeholder_2', 'placeholder_2_data'), ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},
                             'sslice_2': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(0, 54, 1)]),
                                 'shrink_axis_mask': np.array([False, False, False, False, False]),
                                 'new_axis_mask': np.array([False, True, False, False, False])},
                             'sslice_2_data': {'shape': np.array([1, 1, 227, 227, 54])}
                             })
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_reshape,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'sslice_2'),
                                 ('placeholder_begin_data', 'sslice_2'),
                                 ('placeholder_end_data', 'sslice_2'),
                                 ('placeholder_stride_data', 'sslice_2'),
                                 ('sslice_2', 'sslice_2/Reshape_new_data'),
                                 ('sslice_2/Reshape_new_data', 'sslice_2/Reshape_new'),
                                 ('sslice_2/Reshape_new', 'sslice_2_data'),
                                 ('sslice_2_data', 'placeholder_2'),
                                 ('placeholder_2', 'placeholder_2_data')],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},
                                 'sslice_2': {'slices': np.array(
                                     [slice(0, 1, 1), slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1),
                                      slice(0, 54, 1)]),
                                     'shrink_axis_mask': np.array([False, False, False, False, False]),
                                     'new_axis_mask': np.array([False, False, False, False, False])},
                                 'sslice_2_data': {'shape': np.array([1, 1, 227, 227, 54])},
                                 'sslice_2/Reshape_new': {'dim': np.array([1, 1, 227, 227, 54])},
                                 'sslice_2/Reshape_new_data': {'shape': np.array([1, 227, 227, 54])},
                                 })

        pattern = AddReshapeAfterStridedSlice()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'sslice_2_data', check_op_attrs=True)
        graph.clear()
        graph_ref.clear()
        self.assertTrue(flag, resp)

    def test_ss_shrink_new(self):
        graph = build_graph(nodes_attributes_test,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_2'),
                             ('placeholder_begin_data', 'sslice_2'),
                             ('placeholder_end_data', 'sslice_2'),
                             ('placeholder_stride_data', 'sslice_2'),
                             ('sslice_2', 'sslice_2_data'),
                             ('sslice_2_data', 'placeholder_2'),
                             ('placeholder_2', 'placeholder_2_data'), ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},
                             'sslice_2': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 1, 1), slice(0, 227, 1), slice(0, 1, 1), slice(0, 54, 1)]),
                                 'shrink_axis_mask': np.array([False, False, False, True, False]),
                                 'new_axis_mask': np.array([False, True, False, False, False])},
                             'sslice_2_data': {'shape': np.array([1, 1, 227, 54]), 'is_output': True}
                             })
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_reshape,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'sslice_2'),
                                 ('placeholder_begin_data', 'sslice_2'),
                                 ('placeholder_end_data', 'sslice_2'),
                                 ('placeholder_stride_data', 'sslice_2'),
                                 ('sslice_2', 'sslice_2/Reshape_new_data'),
                                 ('sslice_2/Reshape_new_data', 'sslice_2/Reshape_new'),
                                 ('sslice_2/Reshape_new', 'sslice_2/Reshape_shrink_data'),
                                 ('sslice_2/Reshape_shrink_data', 'sslice_2/Reshape_shrink'),
                                 ('sslice_2/Reshape_shrink', 'sslice_2_data'),
                                 ('sslice_2_data', 'placeholder_2'),
                                 ('placeholder_2', 'placeholder_2_data')],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},
                                 'sslice_2': {'slices': np.array(
                                     [slice(0, 1, 1), slice(0, 1, 1), slice(0, 227, 1), slice(0, 1, 1),
                                      slice(0, 54, 1)]),
                                     'shrink_axis_mask': np.array([False, False, False, False, False]),
                                     'new_axis_mask': np.array([False, False, False, False, False])},
                                 'sslice_2_data': {'shape': np.array([1, 1, 227, 54])},
                                 'sslice_2/Reshape_new': {'dim': np.array([1, 1, 227, 1, 54])},
                                 'sslice_2/Reshape_new_data': {'shape': np.array([1, 227, 1, 54])},
                                 'sslice_2/Reshape_shrink': {'dim': np.array([1, 1, 227, 54])},
                                 'sslice_2/Reshape_shrink_data': {'shape': np.array([1, 1, 227, 1, 54])},
                                 })

        pattern = AddReshapeAfterStridedSlice()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'sslice_2_data', check_op_attrs=True)
        graph.clear()
        graph_ref.clear()
        self.assertTrue(flag, resp)


if __name__ == '__main__':
    unittest.main()
