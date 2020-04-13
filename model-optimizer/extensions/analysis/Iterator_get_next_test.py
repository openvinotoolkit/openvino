"""
 Copyright (c) 2020 Intel Corporation

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

from extensions.analysis.inputs import InputsAnalysis
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.unittest.graph import build_graph_with_edge_attrs


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
                      'Run the Model Optimizer with:\n\t\t--input "iter_get_next:0[2 2],iter_get_next:1[1 1]"\n' \
                      'And replace all negative values with positive values'
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
