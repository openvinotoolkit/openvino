# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from extensions.middle.DivisionToZeroFP16Resolver import DivisionToZeroFP16Resolver
from mo.front.common.partial_infer.utils import shape_array
from mo.graph.graph import Node
from unit_tests.utils.graph import build_graph, result, regular_op_with_empty_data, connect, shaped_parameter, \
    valued_const_with_data


class ChangeOutputTypeAttributesTests(unittest.TestCase):

    def test_division_maximum(self):
        self.build_and_test_division_graph(eps=np.array(1e-12), pow_value=np.array(-1), preventing_type='Maximum')

    def test_division_add(self):
        self.build_and_test_division_graph(eps=np.array(1e-12), pow_value=np.array(-1), preventing_type='Add')

    def test_division_arbitrary_negative_pow_1(self):
        self.build_and_test_division_graph(eps=np.array(1e-12), pow_value=np.array(-1/2), preventing_type='Add')

    def test_division_arbitrary_negative_pow_2(self):
        self.build_and_test_division_graph(eps=np.array(1e-12), pow_value=np.array(-0.2), preventing_type='Add')

    def test_division_eps_as_array_1(self):
        self.build_and_test_division_graph(eps=np.array([1e-12, 1e-12]), pow_value=np.array(-1), preventing_type='Add')

    def test_division_eps_as_array_2(self):
        self.build_and_test_division_graph(eps=np.array([1e-12]), pow_value=np.array(-1), preventing_type='Add')

    # in this case graph should not be changed so eps will be left unchanged 1e-2
    # in that case build_and_test_division_graph will raise AssertionError and it's expected
    def test_division_graph_not_changed_1(self):
        try:
            self.build_and_test_division_graph(eps=np.array(1e-2), pow_value=np.array(-1), preventing_type='Maximum')
            raise Exception
        except AssertionError:
            pass

    # if at least one value is greater than FP16 smallest normal value
    # graph should not be changed so eps will be left unchanged ([1e-2, 1e-12])
    # in that case build_and_test_division_graph will raise AssertionError and it's expected
    def test_division_graph_not_changed_2(self):
        try:
            self.build_and_test_division_graph(eps=np.array([1e-2, 1e-12]), pow_value=np.array(-1), preventing_type='Maximum')
            raise Exception
        except AssertionError:
            pass

    def build_and_test_division_graph(self, eps, pow_value, preventing_type):
        nodes = {
            **shaped_parameter('input_1', shape_array((1, 3, 10, 10))),
            **shaped_parameter('input_2', shape_array((1, 3, 10, 10))),
            **regular_op_with_empty_data(preventing_type, {'type': preventing_type, 'op': preventing_type}),
            **regular_op_with_empty_data('negative_pow', {'type': 'Pow', 'op': 'Pow'}),
            **regular_op_with_empty_data('mul', {'type': 'Mul', 'op': 'Mul'}),

            **valued_const_with_data('negative_pow_const', pow_value),
            **valued_const_with_data('eps', eps),
            **result('res'),
        }

        edges = [
            *connect('input_2', '0:' + preventing_type),
            *connect('eps', '1:' + preventing_type),
            *connect(preventing_type, '0:negative_pow'),
            *connect('negative_pow_const', '1:negative_pow'),
            *connect('negative_pow', '1:mul'),
            *connect('input_1', '0:mul'),
            *connect('mul', 'res'),
        ]
        graph = build_graph(nodes, edges)
        graph.graph['cmd_params'].compress_fp16 = True

        DivisionToZeroFP16Resolver().find_and_replace_pattern(graph)

        self.assertTrue(np.all(Node(graph, 'eps').value == np.finfo(np.float16).tiny))

    def test_l2_norm(self):
        nodes = {
            **shaped_parameter('input', shape_array((1, 3, 10, 10))),
            **regular_op_with_empty_data('square', {'type': 'Pow', 'op': 'Pow'}),
            **regular_op_with_empty_data('sum', {'type': 'ReduceSum', 'op': 'ReduceSum'}),
            **regular_op_with_empty_data('max', {'type': 'Maximum', 'op': 'Maximum'}),
            **regular_op_with_empty_data('rsqrt', {'type': 'Pow', 'op': 'Pow'}),
            **regular_op_with_empty_data('l2norm', {'type': 'Mul', 'op': 'Mul'}),

            **valued_const_with_data('rsqrt_pow_const', np.array(-1 / 2)),
            **valued_const_with_data('square_pow', np.array(2)),
            **valued_const_with_data('eps', np.array(1e-12)),
            **result('res'),
        }

        edges = [
            *connect('input:0', '0:square'),
            *connect('square_pow', '1:square'),
            *connect('square', 'sum'),
            *connect('sum', '0:max'),
            *connect('eps', '1:max'),
            *connect('max', '0:rsqrt'),
            *connect('rsqrt_pow_const', '1:rsqrt'),
            *connect('rsqrt', '0:l2norm'),
            *connect('input:0', '1:l2norm', skip_data=True),
            *connect('l2norm', 'res'),
        ]
        graph = build_graph(nodes, edges)
        graph.graph['cmd_params'].compress_fp16 = True

        DivisionToZeroFP16Resolver().find_and_replace_pattern(graph)

        self.assertTrue(np.all(Node(graph, 'eps').value == np.finfo(np.float16).tiny))
