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

from extensions.middle.UselessStridedSlice import UselessStridedSliceEraser
from mo.front.common.partial_infer.utils import int64_array
from mo.middle.passes.eliminate import shape_inference
from mo.utils.unittest.graph import build_graph
from mo.utils.ir_engine.compare_graphs import compare_graphs

nodes_attributes = {
    # input data
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {'value': None, 'shape': int64_array([4, 1, 6]), 'kind': 'data', 'data_type': None},
    #
    'strided_slice': {'type': 'StridedSlice', 'kind': 'op', 'op': 'StridedSlice',
                      'shrink_axis_mask': int64_array([0, 0, 0]), 'new_axis_mask': int64_array([0, 0, 0]),
                      'slices': [slice(0, 4, 1), slice(0, 1, 1), slice(0, 6, 1)]},
    'strided_slice_data': {'value': None, 'shape': int64_array([4, 1, 6]), 'kind': 'data'},
    'strided_slice_input_1_data': {'value': None, 'shape': int64_array([3]), 'kind': 'data'},
    'strided_slice_input_2_data': {'value': None, 'shape': int64_array([3]), 'kind': 'data'},
    'strided_slice_input_3_data': {'value': None, 'shape': int64_array([3]), 'kind': 'data'},
    #
    'strided_slice_2': {'type': 'StridedSlice', 'kind': 'op', 'op': 'StridedSlice',
                        'shrink_axis_mask': int64_array([0, 0, 0]), 'new_axis_mask': int64_array([0, 0, 0]),
                        'slices': [slice(0, 4, 1), slice(0, 1, 1), slice(0, 6, 1)]},
    'strided_slice_2_data': {'value': None, 'shape': int64_array([4, 1, 6]), 'kind': 'data'},
    # Output operation
    'output_op': {'kind': 'op', 'op': 'Result'},
    # squeeze op
    'squeeze': {'type': 'Squeeze', 'kind': 'op', 'op': 'Squeeze'},
    'squeeze_const': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': int64_array([1])},
    'squeeze_const_data': {'kind': 'data'},
    # unsqueeze op
    'unsqueeze': {'type': None, 'kind': 'op', 'op': 'Unsqueeze'},
    'unsqueeze_const': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': int64_array([1])},
    'unsqueeze_const_data': {'kind': 'data'},
    'unsqueeze_data': {'value': None, 'shape': int64_array([4, 6]), 'kind': 'data'},
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

        UselessStridedSliceEraser().find_and_replace_pattern(graph)
        shape_inference(graph)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder', 'placeholder_data'),
                                 ('placeholder_data', 'output_op'),
                                 ],
                                {'placeholder_data': {'shape': int64_array([4, 1, 6])}},
                                nodes_with_edges_only=True
                                )
        (flag, resp) = compare_graphs(graph, graph_ref, 'output_op', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_single_stride_slice_with_shrink_removal(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder', 'placeholder_data'),
                             ('placeholder_data', 'strided_slice'),
                             ('strided_slice_input_1_data', 'strided_slice'),
                             ('strided_slice_input_2_data', 'strided_slice'),
                             ('strided_slice_input_3_data', 'strided_slice'),
                             ('strided_slice', 'strided_slice_data'),
                             ('strided_slice_data', 'output_op'),
                             ],
                            {'strided_slice': {'shrink_axis_mask': int64_array([0, 1, 0])},
                             'strided_slice_data': {'shape': int64_array([4, 6])}},
                            nodes_with_edges_only=True
                            )
        graph.graph['layout'] = 'NCHW'

        UselessStridedSliceEraser().find_and_replace_pattern(graph)
        shape_inference(graph)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder', 'placeholder_data'),
                                 ('placeholder_data', 'squeeze'),
                                 ('squeeze_const', 'squeeze_const_data'),
                                 ('squeeze_const_data', 'squeeze'),
                                 ('squeeze', 'strided_slice_data'),
                                 ('strided_slice_data', 'output_op')
                                 ],
                                {'placeholder_data': {'shape': int64_array([4, 1, 6])},
                                 'strided_slice_data': {'shape': int64_array([4, 6])}},
                                nodes_with_edges_only=True
                                )
        (flag, resp) = compare_graphs(graph, graph_ref, 'output_op', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_single_stride_slice_with_new_removal(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder', 'placeholder_data'),
                             ('placeholder_data', 'strided_slice'),
                             ('strided_slice_input_1_data', 'strided_slice'),
                             ('strided_slice_input_2_data', 'strided_slice'),
                             ('strided_slice_input_3_data', 'strided_slice'),
                             ('strided_slice', 'strided_slice_data'),
                             ('strided_slice_data', 'output_op'),
                             ],
                            {'strided_slice': {'new_axis_mask': int64_array([0, 1, 0, 0])},
                             'strided_slice_data': {'shape': int64_array([4, 1, 1, 6])}},
                            nodes_with_edges_only=True
                            )
        graph.graph['layout'] = 'NCHW'

        UselessStridedSliceEraser().find_and_replace_pattern(graph)
        shape_inference(graph)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder', 'placeholder_data'),
                                 ('placeholder_data', 'unsqueeze'),
                                 ('unsqueeze_const', 'unsqueeze_const_data'),
                                 ('unsqueeze_const_data', 'unsqueeze'),
                                 ('unsqueeze', 'strided_slice_data'),
                                 ('strided_slice_data', 'output_op')
                                 ],
                                {'placeholder_data': {'shape': int64_array([4, 1, 6])},
                                 'strided_slice_data': {'shape': int64_array([4, 1, 1, 6])}},
                                nodes_with_edges_only=True
                                )
        (flag, resp) = compare_graphs(graph, graph_ref, 'output_op', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_single_stride_slice_with_shrink_and_new_removal(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder', 'placeholder_data'),
                             ('placeholder_data', 'strided_slice'),
                             ('strided_slice_input_1_data', 'strided_slice'),
                             ('strided_slice_input_2_data', 'strided_slice'),
                             ('strided_slice_input_3_data', 'strided_slice'),
                             ('strided_slice', 'strided_slice_data'),
                             ('strided_slice_data', 'output_op'),
                             ],
                            {'strided_slice': {'shrink_axis_mask': int64_array([0, 1, 0, 0]),
                                               'new_axis_mask': int64_array([0, 0, 1, 0])},
                             'strided_slice_data': {'shape': int64_array([4, 1, 6])}},
                            nodes_with_edges_only=True
                            )
        graph.graph['layout'] = 'NCHW'

        UselessStridedSliceEraser().find_and_replace_pattern(graph)
        shape_inference(graph)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder', 'placeholder_data'),
                                 ('placeholder_data', 'unsqueeze'),
                                 ('unsqueeze_const', 'unsqueeze_const_data'),
                                 ('unsqueeze_const_data', 'unsqueeze'),
                                 ('unsqueeze', 'unsqueeze_data'),
                                 ('unsqueeze_data', 'squeeze'),
                                 ('squeeze_const', 'squeeze_const_data'),
                                 ('squeeze_const_data', 'squeeze'),
                                 ('squeeze', 'strided_slice_data'),
                                 ('strided_slice_data', 'output_op')
                                 ],
                                {'placeholder_data': {'shape': int64_array([4, 1, 6])},
                                 'unsqueeze_data': {'shape': int64_array([4, 1, 1, 6])},
                                 'strided_slice_data': {'shape': int64_array([4, 1, 6])},
                                 'unsqueeze_const': {'value': int64_array([2])},
                                 },
                                nodes_with_edges_only=True
                                )
        (flag, resp) = compare_graphs(graph, graph_ref, 'output_op', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_single_stride_slice_with_new_and_shrink_removal(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder', 'placeholder_data'),
                             ('placeholder_data', 'strided_slice'),
                             ('strided_slice_input_1_data', 'strided_slice'),
                             ('strided_slice_input_2_data', 'strided_slice'),
                             ('strided_slice_input_3_data', 'strided_slice'),
                             ('strided_slice', 'strided_slice_data'),
                             ('strided_slice_data', 'output_op'),
                             ],
                            {'strided_slice': {'shrink_axis_mask': int64_array([0, 0, 1, 0]),
                                               'new_axis_mask': int64_array([0, 1, 0, 0])},
                             'strided_slice_data': {'shape': int64_array([4, 1, 6])}},
                            nodes_with_edges_only=True
                            )
        graph.graph['layout'] = 'NCHW'

        UselessStridedSliceEraser().find_and_replace_pattern(graph)
        shape_inference(graph)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder', 'placeholder_data'),
                                 ('placeholder_data', 'unsqueeze'),
                                 ('unsqueeze_const', 'unsqueeze_const_data'),
                                 ('unsqueeze_const_data', 'unsqueeze'),
                                 ('unsqueeze', 'unsqueeze_data'),
                                 ('unsqueeze_data', 'squeeze'),
                                 ('squeeze_const', 'squeeze_const_data'),
                                 ('squeeze_const_data', 'squeeze'),
                                 ('squeeze', 'strided_slice_data'),
                                 ('strided_slice_data', 'output_op')
                                 ],
                                {'unsqueeze_data': {'shape': int64_array([4, 1, 1, 6])},
                                 'strided_slice_data': {'shape': int64_array([4, 1, 6])},
                                 'squeeze_const': {'value': int64_array([2])},
                                 },
                                nodes_with_edges_only=True
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

        UselessStridedSliceEraser().find_and_replace_pattern(graph)
        shape_inference(graph)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder', 'placeholder_data'),
                                 ('placeholder_data', 'output_op'),
                                 ],
                                {'placeholder_data': {'shape': int64_array([4, 1, 6])}}
                                )
        (flag, resp) = compare_graphs(graph, graph_ref, 'output_op', check_op_attrs=True)
        self.assertTrue(flag, resp)
