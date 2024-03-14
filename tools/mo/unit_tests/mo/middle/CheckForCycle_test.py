# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.middle.CheckForCycle import CheckForCycle
from openvino.tools.mo.utils.error import Error
from unit_tests.mo.unit_test_with_mocked_telemetry import UnitTestWithMockedTelemetry
from unit_tests.utils.graph import build_graph

nodes_attributes = {'node_1': {'type': 'Identity', 'value': None, 'kind': 'op'},
                    'node_1_data': {'value': None, 'kind': 'data', 'data_type': None},
                    'node_2': {'type': 'Identity', 'value': None, 'kind': 'op'},
                    'concat': {'type': 'Concat', 'value': None, 'kind': 'op'},
                    'node_3': {'type': 'Identity', 'value': None, 'kind': 'op'},
                    'node_3_data': {'value': None, 'kind': 'data', 'data_type': None},
                    # Placeholders
                    'placeholder_1': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
                    'placeholder_2': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                    'pl_1': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                    'pl_1_data': {'value': None, 'kind': 'data', 'data_type': None},
                    'pl_2': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                    'pl_2_data': {'value': None, 'kind': 'data', 'data_type': None},
                    'placeholder_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
                    # ScaleShift layer
                    'scaleshift_1': {'type': 'ScaleShift', 'kind': 'op', 'op': 'ScaleShift'},
                    'scaleshift_1_w': {'value': None, 'shape': None, 'kind': 'data'},
                    'scaleshift_1_b': {'value': None, 'shape': None, 'kind': 'data'},
                    'scaleshift_1_data': {'value': None, 'shape': None, 'kind': 'data'},
                    # Mul op
                    'mul_1': {'type': None, 'kind': 'op', 'op': 'Mul'},
                    'mul_1_w': {'value': None, 'shape': None, 'kind': 'data'},
                    'mul_1_data': {'value': None, 'shape': None, 'kind': 'data'},
                    'op_output': {'kind': 'op', 'op': 'Result', 'infer': lambda x: None}
                    }


class CycleTest(UnitTestWithMockedTelemetry):
    def test_check_for_cycle1(self):
        # cyclic case
        graph = build_graph(nodes_attributes,
                            [('node_1', 'node_1_data'),
                             ('node_1_data', 'node_3'),
                             ('node_3', 'node_3_data'),
                             ('node_3_data', 'node_1')],
                            nodes_with_edges_only=True)
        with self.assertRaisesRegex(Error, 'Graph contains a cycle. Can not proceed.*'):
            CheckForCycle().find_and_replace_pattern(graph)

    def test_check_for_cycle2(self):
        # acyclic case
        graph = build_graph(nodes_attributes,
                            [('node_1', 'node_1_data'),
                             ('node_1_data', 'node_3'),
                             ('node_3', 'node_3_data'),
                             ('node_3_data', 'mul_1'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data')
                             ],
                            nodes_with_edges_only=True)
        try:
            CheckForCycle().find_and_replace_pattern(graph)
        except Error:
            self.fail("Unexpected Error raised")
