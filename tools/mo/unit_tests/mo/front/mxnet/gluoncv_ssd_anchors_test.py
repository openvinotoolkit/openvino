# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import unittest

from openvino.tools.mo.front.mxnet.gluoncv_ssd_anchors import SsdAnchorsReplacer
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

nodes_attributes = {
    'slice_like': {'kind': 'op', 'op': 'slice_like'},
    'model_reshape0': {'kind': 'op', 'op': 'Reshape'},
    'model_reshape0_const': {'kind': 'op', 'op': 'Const', 'value': int64_array([1, -1, 4])},
    'model_reshape1': {'kind': 'op', 'op': 'Reshape'},
    'model_reshape1_const': {'kind': 'op', 'op': 'Const', 'value': int64_array([1, -1, 4])},
    'model_reshape2': {'kind': 'op', 'op': 'Reshape'},
    'model_reshape2_const': {'kind': 'op', 'op': 'Const', 'value': int64_array([1, -1])},
    'reshape0': {'kind': 'op', 'op': 'Reshape'},
    'reshape0_const': {'kind': 'op', 'op': 'Const', 'value': int64_array([1, -1])},
    'concat': {'kind': 'op', 'op': 'Concat'},
    'reshape1': {'kind': 'op', 'op': 'Reshape'},
    'reshape1_const': {'kind': 'op', 'op': 'Const', 'value': int64_array([1, 2, -1])},
    'split': {'kind': 'op', 'op': 'Split', 'num_splits': 2},
    'split_const': {'kind': 'op', 'op': 'Const', 'value': int64_array(1)},
    'reshape2': {'kind': 'op', 'op': 'Reshape'},
    'reshape2_const': {'kind': 'op', 'op': 'Const', 'value': int64_array([-1, 4])},
    'value': {'kind': 'op', 'op': 'Split', 'num_splits': 4},
    'value_const': {'kind': 'op', 'op': 'Const', 'value': int64_array(1)},
    'div_1': {'kind': 'op', 'op': 'Div'},
    'div_1_const': {'kind': 'op', 'op': 'Const', 'value': np.array([2], dtype=np.float32)},
    'div_2': {'kind': 'op', 'op': 'Div'},
    'div_2_const': {'kind': 'op', 'op': 'Const', 'value': np.array([2], dtype=np.float32)},
    'xmin': {'kind': 'op', 'op': 'Sub'},
    'ymin': {'kind': 'op', 'op': 'Sub'},
    'xmax': {'kind': 'op', 'op': 'Add'},
    'ymax': {'kind': 'op', 'op': 'Add'},
    'concat_value': {'kind': 'op', 'op': 'Concat', 'axis': 1},
    'reshape3': {'kind': 'op', 'op': 'Reshape'},
    'reshape3_const': {'kind': 'op', 'op': 'Const', 'value': int64_array([1, 1, -1])},
    'end_concat': {'kind': 'op', 'op': 'Concat'},
    'detection_output': {'kind': 'op', 'op': 'DetectionOutput'}
}


class SsdAnchorsReplacerTest(unittest.TestCase):

    def test_replacer(self):
        graph = build_graph(
            nodes_attrs=nodes_attributes,
            edges=[
                ('slice_like', 'model_reshape0', {'in': 0}),
                ('model_reshape0_const', 'model_reshape0', {'in': 1}),
                ('model_reshape0', 'model_reshape1', {'in': 0}),
                ('model_reshape1_const', 'model_reshape1', {'in': 1}),
                ('model_reshape1', 'model_reshape2', {'in': 0}),
                ('model_reshape2_const', 'model_reshape2', {'in': 1}),
                ('model_reshape2', 'reshape0', {'in': 0}),
                ('reshape0_const', 'reshape0', {'in': 1}),
                ('reshape0', 'concat'),
                ('concat', 'detection_output', {'in': 2})
            ],
            nodes_with_edges_only=True
        )

        ref_graph = build_graph(
            nodes_attrs=nodes_attributes,
            edges=[
                ('slice_like', 'model_reshape0', {'in': 0}),
                ('model_reshape0_const', 'model_reshape0', {'in': 1}),
                ('model_reshape0', 'model_reshape1', {'in': 0}),
                ('model_reshape1_const', 'model_reshape1', {'in': 1}),
                ('model_reshape1', 'model_reshape2', {'in': 0}),
                ('model_reshape2_const', 'model_reshape2', {'in': 1}),
                ('model_reshape2', 'reshape0', {'in': 0}),
                ('reshape0_const', 'reshape0', {'in': 1}),
                ('reshape0', 'concat'),
                ('concat', 'reshape1', {'in': 0}),
                ('reshape1_const', 'reshape1', {'in': 1}),
                ('reshape1', 'split', {'in': 0}),
                ('split_const', 'split', {'in': 1}),
                ('split', 'reshape2', {'out': 0, 'in': 0}),
                ('reshape2_const', 'reshape2', {'in': 1}),
                ('reshape2', 'value', {'in': 0}),
                ('value_const', 'value', {'in': 1}),
                ('value', 'xmin', {'out': 0, 'in': 0}),
                ('value', 'ymin', {'out': 1, 'in': 0}),
                ('value', 'xmax', {'out': 0, 'in': 1}),
                ('value', 'ymax', {'out': 1, 'in': 1}),
                ('value', 'div_1', {'out': 2, 'in': 0}),
                ('value', 'div_2', {'out': 3, 'in': 0}),
                ('div_1_const', 'div_1', {'in': 1}),
                ('div_2_const', 'div_2', {'in': 1}),
                ('div_1', 'xmin', {'in': 1, 'out': 0}),
                ('div_1', 'xmax', {'in': 0, 'out': 0}),
                ('div_2', 'ymin', {'in': 1, 'out': 0}),
                ('div_2', 'ymax', {'in': 0, 'out': 0}),
                ('xmin', 'concat_value', {'in': 0}),
                ('ymin', 'concat_value', {'in': 1}),
                ('xmax', 'concat_value', {'in': 2}),
                ('ymax', 'concat_value', {'in': 3}),
                ('concat_value', 'reshape3', {'in': 0}),
                ('reshape3_const', 'reshape3', {'in': 1}),
                ('reshape3', 'end_concat', {'in': 0}),
                ('split', 'end_concat', {'in': 1}),
                ('end_concat', 'detection_output', {'in': 2})
            ],
            update_attributes={
                'concat': {'axis': 1}
            },
            nodes_with_edges_only=True
        )
        graph.stage = 'front'
        graph.graph['cmd_params'].data_type = 'FP32'
        SsdAnchorsReplacer().find_and_replace_pattern(graph)
        flag, resp = compare_graphs(graph, ref_graph, 'detection_output', check_op_attrs=True)
        self.assertTrue(flag, resp)
