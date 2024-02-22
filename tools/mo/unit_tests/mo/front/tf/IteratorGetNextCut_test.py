# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.tf.IteratorGetNextCut import IteratorGetNextCut
from openvino.tools.mo.front.common.partial_infer.utils import shape_array
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph_with_edge_attrs


class IteratorGetNextAnalysisTest(unittest.TestCase):

    def test_one_output(self):
        graph = build_graph_with_edge_attrs(
            {
                'iter_get_next': {'kind': 'op', 'op': 'IteratorGetNext', 'shapes': shape_array([[2, 2]]),
                                  'types': [np.int32]},
                'sub': {'kind': 'op', 'op': 'Sub'},
            },
            [
                ('iter_get_next', 'sub', {'out': 0, 'in': 0}),
            ]
        )

        graph_ref = build_graph_with_edge_attrs(
            {
                'parameter_1': {'kind': 'op', 'op': 'Parameter', 'shape': shape_array([2, 2]), 'type': np.int32},
                'sub': {'kind': 'op', 'op': 'Sub'},
            },
            [
                ('parameter_1', 'sub', {'out': 0, 'in': 0}),
            ]
        )

        IteratorGetNextCut().find_and_replace_pattern(graph)

        flag, msg = compare_graphs(graph, graph_ref, last_node='sub')
        self.assertTrue(flag, msg)

    def test_two_outputs(self):
        graph = build_graph_with_edge_attrs(
            {
                'iter_get_next': {'kind': 'op', 'op': 'IteratorGetNext', 'shapes': [shape_array([2, 2]),
                                                                                    shape_array([1, 1])],
                                  'types': [np.int32, np.float32]},
                'sub': {'kind': 'op', 'op': 'Sub'},
                'add': {'kind': 'op', 'op': 'Add'},
                'concat': {'kind': 'op', 'op': 'Concat'}
            },
            [
                ('iter_get_next', 'sub', {'out': 0, 'in': 0}),
                ('iter_get_next', 'add', {'out': 1, 'in': 0}),
                ('sub', 'concat', {'out': 0, 'in': 0}),
                ('add', 'concat', {'out': 0, 'in': 1})
            ]
        )

        graph_ref = build_graph_with_edge_attrs(
            {
                'parameter_1': {'kind': 'op', 'op': 'Parameter', 'shape': shape_array([2, 2]), 'data_type': np.int32},
                'parameter_2': {'kind': 'op', 'op': 'Parameter', 'shape': shape_array([1, 1]), 'data_type': np.float32},
                'sub': {'kind': 'op', 'op': 'Sub'},
                'add': {'kind': 'op', 'op': 'Add'},
                'concat': {'kind': 'op', 'op': 'Concat'}
            },
            [
                ('parameter_1', 'sub', {'out': 0, 'in': 0}),
                ('parameter_2', 'add', {'out': 0, 'in': 0}),
                ('sub', 'concat', {'out': 0, 'in': 0}),
                ('add', 'concat', {'out': 0, 'in': 1})
            ]
        )

        IteratorGetNextCut().find_and_replace_pattern(graph)

        flag, msg = compare_graphs(graph, graph_ref, last_node='concat', check_op_attrs=True)
        self.assertTrue(flag, msg)

    def test_unsupported_data_type(self):
        graph = build_graph_with_edge_attrs(
            {
                'iter_get_next': {'kind': 'op', 'op': 'IteratorGetNext', 'shapes': [shape_array([2, 2])],
                                  'types': [None]},
                'sub': {'kind': 'op', 'op': 'Sub'},
                'result': {'kind': 'op', 'op': 'Result'}
            },
            [
                ('iter_get_next', 'sub', {'out': 0, 'in': 0}),
                ('sub', 'result', {'out': 0, 'in': 0}),
            ]
        )

        self.assertRaises(Error, IteratorGetNextCut().find_and_replace_pattern, graph)
