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

from extensions.middle.PixelLinkReshape import PixelLinkReshape
from mo.utils.unittest.graph import build_graph, compare_graphs

nodes_attributes = {
    'placeholder_1': {'shape': None, 'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # Reshape layers
    'reshape_pack': {'type': 'Reshape', 'kind': 'op', 'op': 'Reshape'},
    'reshape_pack_data': {'value': None, 'shape': None, 'kind': 'data'},
    'reshape_split': {'type': 'Reshape', 'kind': 'op', 'op': 'Reshape'},
    'reshape_split_data': {'value': None, 'shape': None, 'kind': 'data'},
    'reshape_unpack': {'type': 'Reshape', 'kind': 'op', 'op': 'Reshape'},
    'reshape_unpack_data': {'value': None, 'shape': None, 'kind': 'data'},
    'strided_slice': {'type': 'StridedSlice', 'kind': 'op', 'op': 'StridedSlice'},
    'strided_slice_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Transpose layer
    'reshape_split/Permute_before': {'type': 'Permute', 'kind': 'op', 'op': 'Permute'},
    'reshape_split/Permute_before_data': {'value': None, 'shape': None, 'kind': 'data'},
    'reshape_pack/Permute_after': {'type': 'Permute', 'kind': 'op', 'op': 'Permute'},
    'reshape_pack/Permute_after_data': {'value': None, 'shape': None, 'kind': 'data'},
    # uncoment when strided slice will be enabled
    # 'reshape_unpack/Permute_after_unpack': {'type': 'Permute', 'kind': 'op', 'op': 'Permute'},
    # 'reshape_unpack/Permute_after_unpack_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Softmax layer
    'softmax_1': {'type': 'SoftMax', 'kind': 'op', 'op': 'SoftMax'},
    'softmax_1_data': {'value': None, 'shape': None, 'kind': 'data'},
}


class ReshapeSoftmaxReshapeTests(unittest.TestCase):
    def test_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'reshape_split'),
                             ('reshape_split', 'reshape_split_data'),
                             ('reshape_split_data', 'reshape_pack'),
                             ('reshape_pack', 'reshape_pack_data'),
                             ('reshape_pack_data', 'softmax_1'),
                             ('softmax_1', 'softmax_1_data'),
                             ('softmax_1_data', 'reshape_unpack'),
                             ('reshape_unpack', 'reshape_unpack_data'),
                             ('reshape_unpack_data', 'strided_slice'),
                             ('strided_slice', 'strided_slice_data'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 16])},
                             'reshape_split': {'dim': np.array([1, 227, 227, 8, 2])},
                             'reshape_split_data': {'shape': np.array([1, 227, 227, 8, 2])},
                             'softmax_1_data': {'shape': np.array([1 * 227 * 227 * 8, 2])},
                             'reshape_pack': {'dim': np.array([1 * 227 * 227 * 8, 2])},
                             'reshape_pack_data': {'shape': np.array([1 * 227 * 227 * 8, 2])},
                             'reshape_unpack': {'dim': np.array([1, 227, 227, 8, 2])},
                             'reshape_unpack_data': {'shape': np.array([1, 227, 227, 8, 2])},
                             'strided_slice': {
                                 'slices': [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(0, 8, 1),
                                            slice(1, 2, 1)],
                                 'shrink_axis_mask': [0, 0, 0, 0, 1],
                                 'new_axis_mask': [0, 0, 0, 0, 0],
                                 'ellipsis_mask': [0, 0, 0, 0, 0],
                                 'begin_mask': [1, 1, 1, 1, 1],
                                 'end_mask': [1, 1, 1, 1, 1], },
                             'strided_slice_data': {'shape': np.array([1, 227, 227, 8])},
                             })
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'reshape_split/Permute_before'),
                                 ('reshape_split/Permute_before', 'reshape_split/Permute_before_data'),
                                 ('reshape_split/Permute_before_data', 'reshape_split'),
                                 ('reshape_split', 'reshape_split_data'),
                                 ('reshape_split_data', 'reshape_pack'),
                                 ('reshape_pack', 'reshape_pack/Permute_after_data'),
                                 ('reshape_pack/Permute_after_data', 'reshape_pack/Permute_after'),
                                 ('reshape_pack/Permute_after', 'reshape_pack_data'),
                                 ('reshape_pack_data', 'softmax_1'),
                                 ('softmax_1', 'softmax_1_data'),
                                 # comment when strided slice will be enabled
                                 ('softmax_1_data', 'strided_slice'),
                                 ('strided_slice', 'reshape_unpack_data'),
                                 ('reshape_unpack_data', 'reshape_unpack'),
                                 ('reshape_unpack', 'strided_slice_data'),
                                 # uncomment when strided slice will be enabled
                                 # ('softmax_1_data', 'reshape_unpack'),
                                 # ('reshape_unpack', 'reshape_unpack/Permute_after_unpack_data'),
                                 # ('reshape_unpack/Permute_after_unpack_data', 'reshape_unpack/Permute_after_unpack'),
                                 # ('reshape_unpack/Permute_after_unpack', 'reshape_unpack_data'),
                                 # ('reshape_unpack_data', 'strided_slice'),
                                 # ('strided_slice', 'strided_slice_data'),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 16])},
                                 'reshape_split/Permute_before_data': {'shape': np.array([1, 227, 16, 227])},
                                 'reshape_split_data': {'shape': np.array([1, 227, 227, 8, 2])},
                                 'reshape_pack_data': {'shape': np.array([1, 2, 1 * 227 * 227 * 8])},
                                 'reshape_pack/Permute_after_data': {'shape': np.array([1, 227 * 227 * 8, 2])},
                                 'softmax_1_data': {'shape': np.array([1, 2, 1 * 227 * 227 * 8])},
                                 # comment when strided slice will be enabled
                                 'reshape_unpack_data': {'shape': np.array([1, 1, 227 * 227 * 8])},
                                 # uncomment when strided slice will be enabled
                                 # 'reshape_unpack_data': {'shape': np.array([1, 8, 227, 227, 2])},
                                 # 'reshape_unpack/Permute_after_unpack_data': {'shape': np.array([1, 227, 227, 8, 2])},
                                 'strided_slice_data': {'shape': np.array([1, 227, 227, 8])}
                                 })

        pattern = PixelLinkReshape()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'strided_slice_data', check_op_attrs=True)
        self.assertTrue(flag, resp)
