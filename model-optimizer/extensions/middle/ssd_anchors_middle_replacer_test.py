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

from extensions.middle.ssd_anchors_middle_replacer import SsdPriorboxReshape, SsdAnchorMiddleReshape, \
    SsdAnchorsMiddleReplacer
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph

nodes_attributes = {
    'slice_like': {'kind': 'op', 'op': 'slice_like'},
    'slice_like_data': {'kind': 'data'},
    'reshape': {'kind': 'op', 'op': 'Reshape'},
    'reshape_data': {'kind': 'data'},
    'reshape_const': {'kind': 'op', 'op': 'Const'},
    'reshape_const_data': {'kind': 'data'},
    'concat': {'kind': 'op', 'op': 'Concat'},
    'concat_data': {'kind': 'data'},

    'strided_slice': {'kind': 'op', 'op': 'StridedSlice'},
    'strided_slice_data': {'kind': 'data'},
    'reshape1': {'kind': 'op', 'op': 'Reshape'},
    'reshape1_data': {'kind': 'data'},
    'reshape1_const': {'kind': 'op', 'op': 'Const'},
    'reshape1_const_data': {'kind': 'data'},
    'reshape2': {'kind': 'op', 'op': 'Reshape'},
    'reshape2_data': {'kind': 'data'},
    'reshape2_const': {'kind': 'op', 'op': 'Const'},
    'reshape2_const_data': {'kind': 'data'},
    'reshape3': {'kind': 'op', 'op': 'Reshape'},
    'reshape3_data': {'kind': 'data'},
    'reshape3_const': {'kind': 'op', 'op': 'Const'},
    'reshape3_const_data': {'kind': 'data'},

    'detection_output': {'kind': 'op', 'op': 'DetectionOutput'},
    'split': {'kind': 'op', 'op': 'Split', 'num_splits': 2},
    'split_data_1': {'kind': 'data'},
    'split_data_2': {'kind': 'data'},
    'split_const': {'kind': 'op', 'op': 'Const'},
    'split_const_data': {'kind': 'data', 'value': int64_array(1)},
    'value': {'kind': 'op', 'op': 'Split', 'num_splits': 4},
    'value_data_1': {'kind': 'data'},
    'value_data_2': {'kind': 'data'},
    'value_data_3': {'kind': 'data'},
    'value_data_4': {'kind': 'data'},
    'value_const': {'kind': 'op', 'op': 'Const'},
    'value_const_data': {'kind': 'data', 'value': int64_array(1)},

    'xmin': {'kind': 'op', 'op': 'Sub'},
    'xmin_data': {'kind': 'data'},
    'ymin': {'kind': 'op', 'op': 'Sub'},
    'ymin_data': {'kind': 'data'},
    'xmax': {'kind': 'op', 'op': 'Add'},
    'xmax_data': {'kind': 'data'},
    'ymax': {'kind': 'op', 'op': 'Add'},
    'ymax_data': {'kind': 'data'},

    'div_1': {'kind': 'op', 'op': 'Div'},
    'div_1_data': {'kind': 'data'},
    'div_1_const': {'kind': 'op', 'op': 'Const'},
    'div_1_const_data': {'kind': 'data', 'value': int64_array([2])},

    'div_2': {'kind': 'op', 'op': 'Div'},
    'div_2_data': {'kind': 'data'},
    'div_2_const': {'kind': 'op', 'op': 'Const'},
    'div_2_const_data': {'kind': 'data', 'value': int64_array([2])},

    'concat_slice_value': {'kind': 'op', 'op': 'Concat', 'axis': 1},
    'concat_slice_value_data':{'kind': 'data'},
    'end_concat': {'kind': 'op', 'op': 'Concat', 'axis': 1},
    'end_concat_data': {'kind': 'data'}
}


class SsdAnchorsMiddleReplacerTest(unittest.TestCase):

    def test_priorbox_reshape(self):
        edges = [
            ('slice_like', 'slice_like_data'),
            ('slice_like_data', 'reshape', {'in': 0}),
            ('reshape_const', 'reshape_const_data'),
            ('reshape_const_data', 'reshape', {'in': 1}),
            ('reshape', 'reshape_data'),
            ('reshape_data', 'concat')
        ]

        graph = build_graph(
            nodes_attrs=nodes_attributes,
            edges=edges,
            nodes_with_edges_only=True,
            update_attributes={
                'reshape_const_data': {'value': int64_array([1, 2, -1])}
            }
        )

        ref_graph = build_graph(
            nodes_attrs=nodes_attributes,
            edges=edges,
            nodes_with_edges_only=True,
            update_attributes={
                'reshape_const_data': {'value': int64_array([1, -1])}
            }
        )

        SsdPriorboxReshape().find_and_replace_pattern(graph)
        flag, resp = compare_graphs(graph, ref_graph, 'concat', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_anchors_reshape(self):
        graph = build_graph(
            nodes_attrs=nodes_attributes,
            edges=[
                ('strided_slice', 'strided_slice_data'),
                ('strided_slice_data', 'reshape'),
                ('reshape', 'reshape_data'),
                ('reshape_data', 'reshape1'),
                ('reshape1', 'reshape1_data'),
                ('reshape1_data', 'reshape2'),
                ('reshape2', 'reshape2_data'),
                ('reshape2_data', 'reshape3'),
                ('reshape3', 'reshape3_data'),
                ('reshape3_data', 'concat')
            ],
            nodes_with_edges_only=True
        )

        ref_graph = build_graph(
            nodes_attrs=nodes_attributes,
            edges=[
                ('strided_slice', 'strided_slice_data'),
                ('strided_slice_data', 'reshape', {'in': 0}),
                ('reshape_const', 'reshape_const_data'),
                ('reshape_const_data', 'reshape', {'in': 1}),
                ('reshape', 'reshape_data'),
                ('reshape_data', 'concat')
            ],
            nodes_with_edges_only=True,
            update_attributes={
                'reshape_const_data': {'value': int64_array([1, -1])}
            }
        )

        SsdAnchorMiddleReshape().find_and_replace_pattern(graph)
        flag, resp = compare_graphs(graph, ref_graph, 'concat', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_anchors_middle_replacer(self):
        graph = build_graph(
            nodes_attrs=nodes_attributes,
            edges=[
                ('strided_slice', 'strided_slice_data'),
                ('strided_slice_data', 'reshape'),
                ('reshape', 'reshape_data'),
                ('reshape_data', 'concat', {'in': 0}),
                ('concat', 'concat_data'),
                ('concat_data', 'detection_output', {'in': 2})
            ]
        )

        ref_graph = build_graph(
            nodes_attrs=nodes_attributes,
            edges=[
                ('strided_slice', 'strided_slice_data'),
                ('strided_slice_data', 'reshape'),
                ('reshape', 'reshape_data'),
                ('reshape_data', 'concat', {'in': 0}),
                ('concat', 'concat_data'),
                ('concat_data', 'reshape1', {'in': 0}),
                ('reshape1_const', 'reshape1_const_data'),
                ('reshape1_const_data', 'reshape1', {'in': 1}),
                ('reshape1', 'reshape1_data'),
                ('reshape1_data', 'split', {'in': 0}),
                ('split_const', 'split_const_data'),
                ('split_const_data', 'split', {'in': 1}),
                ('split', 'split_data_1', {'out': 0}),
                ('split', 'split_data_2', {'out': 1}),
                ('split_data_1', 'reshape2', {'in': 0}),
                ('reshape2_const', 'reshape2_const_data'),
                ('reshape2_const_data', 'reshape2', {'in': 1}),
                ('reshape2', 'reshape2_data'),
                ('reshape2_data', 'value', {'in': 0}),
                ('value_const', 'value_const_data'),
                ('value_const_data', 'value', {'in': 1}),
                ('value', 'value_data_1', {'out': 0}),
                ('value', 'value_data_2', {'out': 1}),
                ('value', 'value_data_3', {'out': 2}),
                ('value', 'value_data_4', {'out': 3}),
                ('div_1_const', 'div_1_const_data'),
                ('div_1_const_data', 'div_1', {'in': 1}),
                ('value_data_3', 'div_1', {'in': 0}),
                ('div_1', 'div_1_data'),
                ('div_2_const', 'div_2_const_data'),
                ('div_2_const_data', 'div_2', {'in': 1}),
                ('value_data_4', 'div_2', {'in': 0}),
                ('div_2', 'div_2_data'),
                ('value_data_1', 'xmin', {'in': 0}),
                ('div_1_data', 'xmin', {'in': 1}),
                ('xmin', 'xmin_data'),
                ('value_data_2', 'ymin', {'in': 0}),
                ('div_2_data', 'ymin', {'in': 1}),
                ('ymin', 'ymin_data'),
                ('div_1_data', 'xmax', {'in': 0}),
                ('value_data_1', 'xmax', {'in': 1}),
                ('xmax', 'xmax_data'),
                ('div_2_data', 'ymax', {'in': 0}),
                ('value_data_2', 'ymax', {'in': 1}),
                ('ymax', 'ymax_data'),
                ('xmin_data', 'concat_slice_value', {'in': 0}),
                ('ymin_data', 'concat_slice_value', {'in': 1}),
                ('xmax_data', 'concat_slice_value', {'in': 2}),
                ('ymax_data', 'concat_slice_value', {'in': 3}),
                ('concat_slice_value', 'concat_slice_value_data'),
                ('concat_slice_value_data', 'reshape3', {'in': 0}),
                ('reshape3_const', 'reshape3_const_data'),
                ('reshape3_const_data', 'reshape3', {'in': 1}),
                ('reshape3', 'reshape3_data'),
                ('reshape3_data', 'end_concat', {'in': 0}),
                ('split_data_2', 'end_concat', {'in': 1}),
                ('end_concat', 'end_concat_data'),
                ('end_concat_data', 'detection_output', {'in': 2})
            ],
            update_attributes={
                'reshape1_const_data': {'value': int64_array([1, 2, -1])},
                'reshape2_const_data': {'value': int64_array([-1, 4])},
                'reshape3_const_data': {'value': int64_array([1, 1, -1])}
            },
            nodes_with_edges_only=True
        )

        SsdAnchorsMiddleReplacer().find_and_replace_pattern(graph)
        flag, resp = compare_graphs(graph, ref_graph, 'detection_output', check_op_attrs=True)
        self.assertTrue(flag, resp)
