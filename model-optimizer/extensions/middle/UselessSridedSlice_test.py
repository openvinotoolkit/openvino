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

import numpy as np

from extensions.middle.UselessStridedSlice import UselessStridedSliceEraser
from mo.utils.unittest.graph import build_graph, compare_graphs

nodes_attributes = {
    # input data
    'placeholder': {'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
    'placeholder_data': {'value': None, 'shape': np.array([4, 5, 6]), 'kind': 'data', 'data_type': None},
    #
    'strided_slice': {'type': 'StridedSlice', 'kind': 'op', 'op': 'StridedSlice', 'shrink_axis_mask': None,
                      'slices': [slice(0, 4, 1), slice(0, 5, 1), slice(0, 6, 1)]},
    'strided_slice_data': {'value': None, 'shape': np.array([4, 5, 6]), 'kind': 'data'},
    'strided_slice_input_1_data': {'value': None, 'shape': np.array([3]), 'kind': 'data'},
    'strided_slice_input_2_data': {'value': None, 'shape': np.array([3]), 'kind': 'data'},
    'strided_slice_input_3_data': {'value': None, 'shape': np.array([3]), 'kind': 'data'},
    #
    'strided_slice_2': {'type': 'StridedSlice', 'kind': 'op', 'op': 'StridedSlice', 'shrink_axis_mask': None,
                        'slices': [slice(0, 4, 1), slice(0, 5, 1), slice(0, 6, 1)]},
    'strided_slice_2_data': {'value': None, 'shape': np.array([4, 5, 6]), 'kind': 'data'},
    # Output operation
    'output_op': {'kind': 'op', 'op': 'OpOutput'},
}


class UselessStridedSliceTests(unittest.TestCase):
    def test_single_stride_slice_removal(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder', 'placeholder_data'),
                             ('placeholder_data', 'strided_slice'),
                             ('strided_slice_input_1_data', 'strided_slice'),
                             ('strided_slice_input_2_data', 'strided_slice'),
                             ('strided_slice_input_3_data', 'strided_slice'),
                             ('strided_slice', 'strided_slice_data'),
                             ('strided_slice_data', 'output_op'),
                             ],
                            {},
                            nodes_with_edges_only=True
                            )

        pattern = UselessStridedSliceEraser()
        pattern.find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder', 'placeholder_data'),
                                 ('placeholder_data', 'output_op'),
                                 ],
                                {'placeholder_data': {'shape': np.array([4, 5, 6])}}
                                )
        (flag, resp) = compare_graphs(graph, graph_ref, 'output_op', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_consecutive_stride_slices_removal(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder', 'placeholder_data'),
                             ('placeholder_data', 'strided_slice'),
                             ('strided_slice_input_1_data', 'strided_slice'),
                             ('strided_slice_input_2_data', 'strided_slice'),
                             ('strided_slice_input_3_data', 'strided_slice'),
                             ('strided_slice', 'strided_slice_data'),
                             ('strided_slice_data', 'strided_slice_2'),
                             ('strided_slice_input_1_data', 'strided_slice_2'),
                             ('strided_slice_input_2_data', 'strided_slice_2'),
                             ('strided_slice_input_3_data', 'strided_slice_2'),
                             ('strided_slice_2', 'strided_slice_2_data'),
                             ('strided_slice_2_data', 'output_op'),
                             ],
                            {},
                            nodes_with_edges_only=True
                            )

        pattern = UselessStridedSliceEraser()
        pattern.find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder', 'placeholder_data'),
                                 ('placeholder_data', 'output_op'),
                                 ],
                                {'placeholder_data': {'shape': np.array([4, 5, 6])}}
                                )
        (flag, resp) = compare_graphs(graph, graph_ref, 'output_op', check_op_attrs=True)
        self.assertTrue(flag, resp)
