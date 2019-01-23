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

from extensions.middle.SliceConverter import ConvertSlice
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph, compare_graphs
from mo.ops.slice import Slice

nodes_attributes = {
    # input data
    'placeholder_1': {'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # Slice layer
    'slice': {'type': 'Slice', 'kind': 'op', 'op': 'Slice'},
    'slice_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Output operation
    'output_op': {'type': 'Const', 'value': None, 'kind': 'op', 'op': 'Const'},
    'output_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # Crop layer
    'crop': {'type': 'Crop', 'kind': 'op', 'op': 'Crop', 'axis': None, 'offset': None, 'dim': None},
    'dim': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # StridedSlice layer
    'strided_slice': {'type': 'StridedSlice', 'kind': 'op', 'op': 'StridedSlice', 'slices': None,
                      'shrink_axis_mask': None}
}


class ConvertSliceTests(unittest.TestCase):
    def test_1(self):
        """
        Testing case with non-constant path and multiple
        slicing dimensions
        :return:
        """
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'slice'),
                             ('slice', 'slice_data'),
                             ('slice_data', 'output_op'),
                             ('output_op', 'output_data')
                             ],
                            {'placeholder_1_data': {'shape': np.array([4, 5, 6])},
                             'slice': {'start': np.array([1, 2, 3]), 'end': np.array([3, 4, 4]), 'axis': None},
                             'output_op': {'is_output': True},
                             }
                            )
        slice_node = Node(graph, 'slice')
        Slice.infer(slice_node)

        pattern = ConvertSlice()
        pattern.find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'crop'),
                                 ('crop', 'slice_data'),
                                 ('slice_data', 'output_op'),
                                 ('output_op', 'output_data')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([4, 5, 6])},
                                 'crop': {'axis': np.array([0, 1, 2]), 'offset': np.array([1, 2, 3]),
                                          },
                                 'output_op': {'is_output': True},
                                 'dim': {'dim': np.array([2, 2, 1])},
                                 }
                                )
        (flag, resp) = compare_graphs(graph, graph_ref, 'output_op', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_2(self):
        """
        Testing case with constant path and one
         slicing dimension
        """
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'slice'),
                             ('slice', 'slice_data'),
                             ('slice_data', 'output_op'),
                             ('output_op', 'output_data')
                             ],
                            {'placeholder_1_data': {'shape': np.array([4, 5, 6])},
                             'slice': {'start': np.array([1]), 'end': np.array([3]), 'axis': None},
                             'output_op': {'is_output': True}
                             }
                            )
        slice_node = Node(graph, 'slice')
        Slice.infer(slice_node)

        pattern = ConvertSlice()
        pattern.find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'strided_slice'),
                                 ('strided_slice', 'slice_data'),
                                 ('slice_data', 'output_op'),
                                 ('output_op', 'output_data')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([4, 5, 6])},
                                 'strided_slice': {'slices': np.array([slice(1, 3, 1),slice(0, 5, 1),slice(0, 6, 1)]),
                                                   'shrink_axis_mask': np.array([False, False, False])},
                                 'output_op': {'is_output': True}
                                 }
                                )

        (flag, resp) = compare_graphs(graph, graph_ref, 'output_op', check_op_attrs=True)
        self.assertTrue(flag, resp)
