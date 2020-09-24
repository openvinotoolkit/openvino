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

from extensions.middle.SliceConverter import ConvertSlice
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.ops.slice import Slice
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph

nodes_attributes = {
    # input data
    'placeholder_1': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_2': {'type': 'Const', 'kind': 'op', 'op': 'Const'},
    'placeholder_3': {'type': 'Const', 'kind': 'op', 'op': 'Const'},
    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_3_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # Slice layer
    'slice': {'type': 'Slice', 'kind': 'op', 'op': 'Slice', 'name': 'slice_node'},
    'slice_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Output operation
    'output_op': {'type': 'Const', 'value': None, 'kind': 'op', 'op': 'Const'},
    'output_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'op_output': { 'kind': 'op', 'op': 'Result'},
    # StridedSlice layer
    'strided_slice': {'kind': 'op', 'op': 'StridedSlice', 'slices': None, 'shrink_axis_mask': None}
}


class ConvertSliceTests(unittest.TestCase):
    nodes_attributes = {
        # input data
        'placeholder_1': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
        'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
        # Slice layer inputs
        'starts': {'type': 'Const', 'kind': 'op', 'op': 'Const'},
        'starts_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
        'ends': {'type': 'Const', 'kind': 'op', 'op': 'Const'},
        'ends_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
        'strides': {'type': 'Const', 'kind': 'op', 'op': 'Const'},
        'strides_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
        'axes': {'type': 'Const', 'kind': 'op', 'op': 'Const'},
        'axes_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
        'steps': {'type': 'Const', 'kind': 'op', 'op': 'Const'},
        'steps_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
        # Slice layer
        'slice': {'type': 'Slice', 'kind': 'op', 'op': 'Slice', 'name': 'slice_node'},
        'slice_data': {'value': None, 'shape': None, 'kind': 'data'},
        # Output operation
        'output_op': {'type': 'Const', 'kind': 'op', 'op': 'Const'},
        'output_data': {'shape': None, 'kind': 'data', 'data_type': None},
        'op_output': {'kind': 'op', 'op': 'Result'},
        # StridedSlice layer
        'strided_slice': {'kind': 'op', 'op': 'StridedSlice', 'slices': None, 'shrink_axis_mask': None}
    }

    def test_slice_all_params(self):
        input_shape = int64_array([5, 10, 20])
        starts_value = int64_array([4, 2])
        ends_value = int64_array([15, 8])
        axes_value = int64_array([2, 1])
        steps_value = int64_array([1, 1])

        masks_value = np.zeros([len(input_shape)], dtype=np.int64)
        graph = build_graph(self.nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'slice', {'in': 0}),
                             ('starts', 'starts_data'),
                             ('starts_data', 'slice', {'in': 1}),
                             ('ends', 'ends_data'),
                             ('ends_data', 'slice', {'in': 2}),
                             ('axes', 'axes_data'),
                             ('axes_data', 'slice', {'in': 3}),
                             ('steps', 'steps_data'),
                             ('steps_data', 'slice', {'in': 4}),
                             ('slice', 'slice_data'),
                             ('slice_data', 'output_op'),
                             ('output_op', 'output_data'),
                             ('output_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': input_shape},
                             'starts': {'shape': starts_value.shape, 'value': starts_value},
                             'starts_data': {'shape': starts_value.shape, 'value': starts_value},
                             'ends': {'shape': ends_value.shape, 'value': ends_value},
                             'ends_data': {'shape': ends_value.shape, 'value': ends_value},
                             'steps': {'shape': steps_value.shape, 'value': steps_value},
                             'steps_data': {'shape': steps_value.shape, 'value': steps_value},
                             'axes': {'shape': axes_value.shape, 'value': axes_value},
                             'axes_data': {'shape': axes_value.shape, 'value': axes_value},
                             }, nodes_with_edges_only=True
                            )
        slice_node = Node(graph, 'slice')
        Slice.infer(slice_node)

        pattern = ConvertSlice()
        pattern.find_and_replace_pattern(graph)

        ss_node = Node(graph, graph.get_node_id_by_name('slice_node'))
        assert ss_node.type == 'StridedSlice', 'Something wrong with transformed Slice node'

        graph_ref = build_graph(self.nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'strided_slice', {'in': 0}),
                                 ('starts', 'starts_data'),
                                 ('starts_data', 'strided_slice', {'in': 1}),
                                 ('ends', 'ends_data'),
                                 ('ends_data', 'strided_slice', {'in': 2}),
                                 ('strides', 'strides_data'),
                                 ('strides_data', 'strided_slice', {'in': 3}),
                                 ('strided_slice', 'slice_data'),
                                 ('slice_data', 'output_op'),
                                 ('output_op', 'output_data'),
                                 ('output_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': input_shape},
                                 'strided_slice': {'new_axis_mask': masks_value, 'shrink_axis_mask': masks_value,
                                                   'ellipsis_mask': masks_value, 'begin_mask': int64_array([0, 1, 1]),
                                                   'end_mask': int64_array([0, 1, 1])},
                                 'slice_data': {'shape': int64_array([5, 6, 11])}
                                 }, nodes_with_edges_only=True
                                )
        (flag, resp) = compare_graphs(graph, graph_ref, 'output_op', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_no_steps_no_axes(self):
        input_shape = int64_array([5, 10, 20])
        starts_value = int64_array([3, 2, 7])
        ends_value = int64_array([5, 8, 15])
        steps_value = int64_array([1, 1, 1])
        masks_value = np.zeros([len(input_shape)], dtype=np.int64)
        graph = build_graph(self.nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'slice', {'in': 0}),
                             ('starts', 'starts_data'),
                             ('starts_data', 'slice', {'in': 1}),
                             ('ends', 'ends_data'),
                             ('ends_data', 'slice', {'in': 2}),
                             ('slice', 'slice_data'),
                             ('slice_data', 'output_op'),
                             ('output_op', 'output_data'),
                             ('output_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': input_shape},
                             'starts': {'shape': starts_value.shape, 'value': starts_value},
                             'starts_data': {'shape': starts_value.shape, 'value': starts_value},
                             'ends': {'shape': ends_value.shape, 'value': ends_value},
                             'ends_data': {'shape': ends_value.shape, 'value': ends_value},
                             }, nodes_with_edges_only=True
                            )
        slice_node = Node(graph, 'slice')
        Slice.infer(slice_node)

        pattern = ConvertSlice()
        pattern.find_and_replace_pattern(graph)

        ss_node = Node(graph, graph.get_node_id_by_name('slice_node'))
        assert ss_node.type == 'StridedSlice', 'Something wrong with transformed Slice node'

        graph_ref = build_graph(self.nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'strided_slice', {'in': 0}),
                                 ('starts', 'starts_data'),
                                 ('starts_data', 'strided_slice', {'in': 1}),
                                 ('ends', 'ends_data'),
                                 ('ends_data', 'strided_slice', {'in': 2}),
                                 ('strides', 'strides_data'),
                                 ('strides_data', 'strided_slice', {'in': 3}),
                                 ('strided_slice', 'slice_data'),
                                 ('slice_data', 'output_op'),
                                 ('output_op', 'output_data'),
                                 ('output_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': input_shape},
                                 'strided_slice': {'new_axis_mask': masks_value, 'shrink_axis_mask': masks_value,
                                                   'ellipsis_mask': masks_value, 'begin_mask': np.ones([3]),
                                                   'end_mask': np.ones([3])},
                                 'slice_data': {'shape': int64_array([2, 6, 8])}
                                 }, nodes_with_edges_only=True
                                )
        (flag, resp) = compare_graphs(graph, graph_ref, 'output_op', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_no_axes(self):
        input_shape = int64_array([5, 10, 20])
        starts_value = int64_array([3, 2, 7])
        ends_value = int64_array([5, 8, 15])
        steps_value = int64_array([2, 3, 1])
        masks_value = np.zeros([len(input_shape)], dtype=np.int64)
        graph = build_graph(self.nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'slice', {'in': 0}),
                             ('starts', 'starts_data'),
                             ('starts_data', 'slice', {'in': 1}),
                             ('ends', 'ends_data'),
                             ('ends_data', 'slice', {'in': 2}),
                             ('steps', 'steps_data'),
                             ('steps_data', 'slice', {'in': 4}),
                             ('slice', 'slice_data'),
                             ('slice_data', 'output_op'),
                             ('output_op', 'output_data'),
                             ('output_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': input_shape},
                             'starts': {'shape': starts_value.shape, 'value': starts_value},
                             'starts_data': {'shape': starts_value.shape, 'value': starts_value},
                             'ends': {'shape': ends_value.shape, 'value': ends_value},
                             'ends_data': {'shape': ends_value.shape, 'value': ends_value},
                             'steps': {'shape': steps_value.shape, 'value': steps_value},
                             'steps_data': {'shape': steps_value.shape, 'value': steps_value},
                             }, nodes_with_edges_only=True
                            )
        slice_node = Node(graph, 'slice')
        Slice.infer(slice_node)

        pattern = ConvertSlice()
        pattern.find_and_replace_pattern(graph)

        ss_node = Node(graph, graph.get_node_id_by_name('slice_node'))
        assert ss_node.type == 'StridedSlice', 'Something wrong with transformed Slice node'

        graph_ref = build_graph(self.nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'strided_slice', {'in': 0}),
                                 ('starts', 'starts_data'),
                                 ('starts_data', 'strided_slice', {'in': 1}),
                                 ('ends', 'ends_data'),
                                 ('ends_data', 'strided_slice', {'in': 2}),
                                 ('strides', 'strides_data'),
                                 ('strides_data', 'strided_slice', {'in': 3}),
                                 ('strided_slice', 'slice_data'),
                                 ('slice_data', 'output_op'),
                                 ('output_op', 'output_data'),
                                 ('output_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': input_shape},
                                 'strided_slice': {'new_axis_mask': masks_value, 'shrink_axis_mask': masks_value,
                                                   'ellipsis_mask': masks_value, 'begin_mask': np.ones([3]),
                                                   'end_mask': np.ones([3])},
                                 'slice_data': {'shape': int64_array([1, 2, 8])}
                                 }, nodes_with_edges_only=True
                                )
        (flag, resp) = compare_graphs(graph, graph_ref, 'output_op', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_no_steps(self):
        input_shape = int64_array([5, 10, 20])
        starts_value = int64_array([4, 2])
        ends_value = int64_array([15, 8])
        axes_value = int64_array([2, 1])
        masks_value = np.zeros([len(input_shape)], dtype=np.int64)
        graph = build_graph(self.nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'slice', {'in': 0}),
                             ('starts', 'starts_data'),
                             ('starts_data', 'slice', {'in': 1}),
                             ('ends', 'ends_data'),
                             ('ends_data', 'slice', {'in': 2}),
                             ('axes', 'axes_data'),
                             ('axes_data', 'slice', {'in': 3}),
                             ('slice', 'slice_data'),
                             ('slice_data', 'output_op'),
                             ('output_op', 'output_data'),
                             ('output_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': input_shape},
                             'starts': {'shape': starts_value.shape, 'value': starts_value},
                             'starts_data': {'shape': starts_value.shape, 'value': starts_value},
                             'ends': {'shape': ends_value.shape, 'value': ends_value},
                             'ends_data': {'shape': ends_value.shape, 'value': ends_value},
                             'axes': {'shape': axes_value.shape, 'value': axes_value},
                             'axes_data': {'shape': axes_value.shape, 'value': axes_value},
                             }, nodes_with_edges_only=True
                            )
        slice_node = Node(graph, 'slice')
        Slice.infer(slice_node)

        pattern = ConvertSlice()
        pattern.find_and_replace_pattern(graph)

        ss_node = Node(graph, graph.get_node_id_by_name('slice_node'))
        assert ss_node.type == 'StridedSlice', 'Something wrong with transformed Slice node'

        graph_ref = build_graph(self.nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'strided_slice', {'in': 0}),
                                 ('starts', 'starts_data'),
                                 ('starts_data', 'strided_slice', {'in': 1}),
                                 ('ends', 'ends_data'),
                                 ('ends_data', 'strided_slice', {'in': 2}),
                                 ('strides', 'strides_data'),
                                 ('strides_data', 'strided_slice', {'in': 3}),
                                 ('strided_slice', 'slice_data'),
                                 ('slice_data', 'output_op'),
                                 ('output_op', 'output_data'),
                                 ('output_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': input_shape},
                                 'strided_slice': {'new_axis_mask': masks_value, 'shrink_axis_mask': masks_value,
                                                   'ellipsis_mask': masks_value, 'begin_mask': int64_array([0, 1, 1]),
                                                   'end_mask': int64_array([0, 1, 1])},
                                 'slice_data': {'shape': int64_array([5, 6, 11])}
                                 }, nodes_with_edges_only=True
                                )
        (flag, resp) = compare_graphs(graph, graph_ref, 'output_op', check_op_attrs=True)
        self.assertTrue(flag, resp)
