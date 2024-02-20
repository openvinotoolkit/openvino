# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.analysis.inputs import InputsAnalysis
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from unit_tests.utils.graph import build_graph_with_edge_attrs


class IteratorGetNextAnalysisTest(unittest.TestCase):

    def test_positive(self):
        graph = build_graph_with_edge_attrs(
            {
                'iter_get_next': {'kind': 'op', 'op': 'IteratorGetNext', 'shapes': int64_array([[2, 2], [1, 1]]),
                                  'types': [None, None]},
                'sub': {'kind': 'op', 'op': 'Sub'},
                'add': {'kind': 'op', 'op': 'Add'}
            },
            [
                ('iter_get_next', 'sub', {'out': 0, 'in': 0}),
                ('iter_get_next', 'add', {'out': 1, 'in': 0})
            ]
        )
        inputs_desc = {}
        message = InputsAnalysis.iterator_get_next_analysis(graph, inputs_desc)
        ref_message = 'It looks like there is IteratorGetNext as input\n' \
                      'Run the Model Optimizer without --input option \n' \
                      'Otherwise, try to run the Model Optimizer with:\n\t\t--input "iter_get_next:0[2 2],iter_get_next:1[1 1]"\n'
        self.assertEqual(message, ref_message)

    def test_negative(self):
        graph = build_graph_with_edge_attrs(
            {
                'placeholder': {'kind': 'op', 'op': 'Parameter'},
                'sub': {'kind': 'op', 'op': 'Sub'},
                'add': {'kind': 'op', 'op': 'Add'}
            },
            [
                ('placeholder', 'sub', {'out': 0, 'in': 0}),
                ('placeholder', 'add', {'out': 0, 'in': 0})
            ]
        )

        inputs_desc = {}
        message = InputsAnalysis.iterator_get_next_analysis(graph, inputs_desc)
        self.assertEqual(message, None)
