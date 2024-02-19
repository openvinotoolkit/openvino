# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.middle.CutInputHavingZeroDimFromConcat import CutInputHavingZeroDimFromConcat
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

node_attrs_for_the_case_when_there_are_no_zero_shape_constants = {
    'const0': {
        'kind': 'op',
        'type': 'Const',
        'op': 'Const',
        'shape': int64_array([1, 2, 5]),
        'value': np.zeros((1, 2, 5))
    },
    'const0_data': {'kind': 'data', 'shape': int64_array([1, 2, 5]), 'value': None},
    'const1': {
        'kind': 'op',
        'type': 'Const',
        'op': 'Const',
        'shape': int64_array([1, 2, 7]),
        'value': np.zeros((1, 2, 7))
    },
    'const1_data': {'kind': 'data', 'shape': int64_array([1, 2, 7]), 'value': None},
    'placeholder': {'kind': 'op', 'type': 'Parameter', 'op': 'Parameter'},
    'placeholder_data': {
        'kind': 'data',
        'value': None,
        'shape': int64_array([1, 2, 8]),
        'data_type': None
    },
    'concat': {'kind': 'op', 'type': 'Concat', 'op': 'Concat', 'axis': 2},
    'concat_data': {'kind': 'data', 'shape': int64_array([1, 2, 20]), 'value': None},
    'output': {'kind': 'op', 'op': 'Result', 'type': 'Result'},
}


edges_for_the_case_when_there_are_no_zero_shape_constants = [
    ('const0', 'const0_data'),
    ('const1', 'const1_data'),
    ('placeholder', 'placeholder_data'),
    ('const0_data', 'concat', {'in': 0}),
    ('const1_data', 'concat', {'in': 1}),
    ('placeholder_data', 'concat', {'in': 2}),
    ('concat', 'concat_data'),
    ('concat_data', 'output')
]


class CutInputHavingZeroDimFromConcatTest(unittest.TestCase):
    """
    This class tests deleting of inputs of Concat having zeros in their shapes, if not all inputs have such shapes.
    """
    def test_when_need_to_do_nothing(self):
        graph = build_graph(
            nodes_attrs=node_attrs_for_the_case_when_there_are_no_zero_shape_constants,
            edges=edges_for_the_case_when_there_are_no_zero_shape_constants
        )
        ref_graph = build_graph(
            nodes_attrs=node_attrs_for_the_case_when_there_are_no_zero_shape_constants,
            edges=edges_for_the_case_when_there_are_no_zero_shape_constants
        )
        CutInputHavingZeroDimFromConcat().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_when_there_are_three_inputs_and_middle_constant_has_zero_in_shape(self):
        graph = build_graph(
            nodes_attrs={
                'const0': {
                    'kind': 'op',
                    'type': 'Const',
                    'op': 'Const',
                    'shape': int64_array([1, 2, 5]),
                    'value': np.zeros((1, 2, 5))
                },
                'const0_data': {'kind': 'data', 'shape': int64_array([1, 2, 5]), 'value': None},
                'const1': {
                    'kind': 'op',
                    'type': 'Const',
                    'op': 'Const',
                    'shape': int64_array([1, 2, 0]),
                    'value': np.zeros((1, 2, 0))
                },
                'const1_data': {'kind': 'data', 'shape': int64_array([1, 2, 0]), 'value': None},
                'placeholder': {'kind': 'op', 'type': 'Parameter', 'op': 'Parameter'},
                'placeholder_data': {
                    'kind': 'data',
                    'value': None,
                    'shape': int64_array([1, 2, 17]),
                    'data_type': None
                },
                'concat': {'kind': 'op', 'type': 'Concat', 'op': 'Concat', 'axis': 2},
                'concat_data': {'kind': 'data', 'shape': int64_array([1, 2, 22]), 'value': None},
                'output': {'kind': 'op', 'op': 'Result', 'type': 'Result'},
            },
            edges=[
                ('const0', 'const0_data'),
                ('const1', 'const1_data'),
                ('placeholder', 'placeholder_data'),
                ('const0_data', 'concat', {'in': 0}),
                ('const1_data', 'concat', {'in': 1}),
                ('placeholder_data', 'concat', {'in': 2}),
                ('concat', 'concat_data'),
                ('concat_data', 'output')
            ]
        )
        ref_graph = build_graph(
            nodes_attrs={
                'const0': {
                    'kind': 'op',
                    'type': 'Const',
                    'op': 'Const',
                    'shape': int64_array([1, 2, 5]),
                    'value': np.zeros((1, 2, 5))
                },
                'const0_data': {'kind': 'data', 'shape': int64_array([1, 2, 5]), 'value': None},
                'placeholder': {'kind': 'op', 'type': 'Parameter', 'op': 'Parameter'},
                'placeholder_data': {
                    'kind': 'data',
                    'value': None,
                    'shape': int64_array([1, 2, 17]),
                    'data_type': None
                },
                'concat': {'kind': 'op', 'type': 'Concat', 'op': 'Concat', 'axis': 2},
                'concat_data': {'kind': 'data', 'shape': int64_array([1, 2, 22]), 'value': None},
                'output': {'kind': 'op', 'op': 'Result', 'type': 'Result'},
            },
            edges=[
                ('const0', 'const0_data'),
                ('placeholder', 'placeholder_data'),
                ('const0_data', 'concat', {'in': 0}),
                ('placeholder_data', 'concat', {'in': 1}),
                ('concat', 'concat_data'),
                ('concat_data', 'output')
            ]
        )
        CutInputHavingZeroDimFromConcat().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_there_are_four_inputs_and_first_and_third_input_have_zero_in_their_shapes(self):
        graph = build_graph(
            nodes_attrs={
                'const0': {
                    'kind': 'op',
                    'type': 'Const',
                    'op': 'Const',
                    'shape': int64_array([5, 0]),
                    'value': np.zeros((5, 0))
                },
                'const0_data': {'kind': 'data', 'shape': int64_array([5, 0]), 'value': None},
                'placeholder': {'kind': 'op', 'type': 'Parameter', 'op': 'Parameter'},
                'placeholder_data': {
                    'kind': 'data',
                    'value': None,
                    'shape': int64_array([5, 17]),
                    'data_type': None
                },
                'const2': {
                    'kind': 'op',
                    'type': 'Const',
                    'op': 'Const',
                    'shape': int64_array([5, 0]),
                    'value': np.zeros((5, 0))
                },
                'const2_data': {'kind': 'data', 'shape': int64_array([5, 0]), 'value': None},
                'const3': {
                    'kind': 'op',
                    'type': 'Const',
                    'op': 'Const',
                    'shape': int64_array([5, 23]),
                    'value': np.zeros((5, 23))
                },
                'const3_data': {'kind': 'data', 'shape': int64_array([5, 23]), 'value': None},
                'concat': {'kind': 'op', 'type': 'Concat', 'op': 'Concat', 'axis': 1},
                'concat_data': {'kind': 'data', 'shape': int64_array([5, 40]), 'value': None},
                'output': {'kind': 'op', 'op': 'Result', 'type': 'Result'},
            },
            edges=[
                ('const0', 'const0_data'),
                ('placeholder', 'placeholder_data'),
                ('const2', 'const2_data'),
                ('const3', 'const3_data'),
                ('const0_data', 'concat', {'in': 0}),
                ('placeholder_data', 'concat', {'in': 1}),
                ('const2_data', 'concat', {'in': 2}),
                ('const3_data', 'concat', {'in': 3}),
                ('concat', 'concat_data'),
                ('concat_data', 'output')
            ]
        )
        ref_graph = build_graph(
            nodes_attrs={
                'placeholder': {'kind': 'op', 'type': 'Parameter', 'op': 'Parameter'},
                'placeholder_data': {
                    'kind': 'data',
                    'value': None,
                    'shape': int64_array([5, 17]),
                    'data_type': None
                },
                'const3': {
                    'kind': 'op',
                    'type': 'Const',
                    'op': 'Const',
                    'shape': int64_array([5, 23]),
                    'value': np.zeros((5, 23))
                },
                'const3_data': {'kind': 'data', 'shape': int64_array([5, 23]), 'value': None},
                'concat': {'kind': 'op', 'type': 'Concat', 'op': 'Concat', 'axis': 1},
                'concat_data': {'kind': 'data', 'shape': int64_array([5, 40]), 'value': None},
                'output': {'kind': 'op', 'op': 'Result', 'type': 'Result'},
            },
            edges=[
                ('placeholder', 'placeholder_data'),
                ('const3', 'const3_data'),
                ('placeholder_data', 'concat', {'in': 0}),
                ('const3_data', 'concat', {'in': 1}),
                ('concat', 'concat_data'),
                ('concat_data', 'output')
            ]
        )
        CutInputHavingZeroDimFromConcat().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)
