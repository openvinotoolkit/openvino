# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import numpy.testing as npt
import unittest

from extensions.ops.If import If
from mo.ops.shape import Shape
from extensions.ops.elementwise import Add, Mul
from extensions.ops.identity import Identity
from extensions.ops.parameter import Parameter
from mo.front.common.partial_infer.concat import concat_infer
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node, Graph
from mo.middle.passes.infer import partial_infer
from mo.ops.concat import Concat
from mo.ops.eltwise import eltwise_infer
from mo.ops.result import Result
from unit_tests.utils.graph import build_graph_with_edge_attrs, build_graph
from unit_tests.utils.graph import regular_op_with_empty_data, connect, const, result, valued_const_with_data, \
    regular_op, empty_data


class TestIf(unittest.TestCase):
    def test_simple_shape_inf(self):
        then_graph_nodes = {**regular_op_with_empty_data('param_1', {'type': 'Parameter', 'kind': 'op', 'input_id': 1,
                                                                     'shape': None, 'infer': Parameter.infer}),
                            **regular_op_with_empty_data('param_2', {'type': 'Parameter', 'kind': 'op', 'input_id': 2,
                                                                     'shape': None, 'infer': Parameter.infer}),
                            **regular_op_with_empty_data('add', {'type': 'Add', 'kind': 'op', 'op': 'Add',
                                                                 'infer': lambda node: eltwise_infer(node,
                                                                                                     Add.operation)}),
                            **regular_op_with_empty_data('mul', {'type': 'Mul', 'kind': 'op', 'op': 'Mul',
                                                                 'infer': lambda node: eltwise_infer(node,
                                                                                                     Mul.operation)}),
                            **regular_op_with_empty_data('res1', {'kind': 'op', 'type': 'Result', 'op': 'Result',
                                                                  'infer': lambda x: 0, 'output_id': 0}),
                            **regular_op_with_empty_data('res2', {'kind': 'op', 'type': 'Result', 'op': 'Result',
                                                                  'infer': lambda x: 0, 'output_id': 1})}
        then_graph_edges = [*connect('param_1', '0:add'),
                            *connect('param_2', '1:add'),
                            *connect('param_1', '1:mul'),
                            *connect('param_2', '0:mul'),
                            *connect('add', 'res1'),
                            *connect('mul', 'res2'),
                            ]

        else_graph_nodes = {**regular_op_with_empty_data('param_1', {'type': 'Parameter', 'kind': 'op', 'input_id': 1,
                                                                     'shape': None, 'infer': Parameter.infer}),
                            **regular_op_with_empty_data('param_2', {'type': 'Parameter', 'kind': 'op', 'input_id': 3,
                                                                     'shape': None, 'infer': Parameter.infer}),
                            **regular_op_with_empty_data('identity',
                                                         {'kind': 'op', 'op': 'Identity', 'infer': Identity.infer}),
                            **regular_op_with_empty_data('identity_1',
                                                         {'kind': 'op', 'op': 'Identity', 'infer': Identity.infer}),
                            **regular_op_with_empty_data('res1', {'kind': 'op', 'type': 'Result', 'op': 'Result',
                                                                  'infer': lambda x: 0, 'output_id': 0}),
                            **regular_op_with_empty_data('res2', {'kind': 'op', 'type': 'Result', 'op': 'Result',
                                                                  'infer': lambda x: 0, 'output_id': 1})}
        else_graph_edges = [*connect('param_1', 'identity'),
                            *connect('param_2', 'identity_1'),
                            *connect('identity_1', 'res2'),
                            *connect('identity', 'res1'), ]
        then_graph = build_graph_with_edge_attrs(then_graph_nodes, then_graph_edges)
        else_graph = build_graph_with_edge_attrs(else_graph_nodes, else_graph_edges)
        external_graph_nodes = {
            **valued_const_with_data('cond', np.array([True], dtype=np.bool)),
            **valued_const_with_data('input_2', int64_array([3, 2, 1])),
            **valued_const_with_data('input_1', int64_array([1, 2, 3])),
            **valued_const_with_data('input_3', int64_array([8, 4])),
            **regular_op('if', {'kind': 'op', 'op': 'If', 'then_graph': then_graph,
                                'else_graph': else_graph, 'infer': If.infer}),
            **empty_data('if_d_1'),
            **empty_data('if_d_2'),
            **result('res_1'),
            **result('res_2')}
        external_graph_edges = [*connect('cond', '0:if'),
                                *connect('input_1', '1:if'),
                                *connect('input_2', '2:if'),
                                *connect('input_3', '3:if'),
                                ('if', 'if_d_1', {'out': 0}),
                                ('if', 'if_d_2', {'out': 1}),
                                ('if_d_1', 'res_1'),
                                ('if_d_2', 'res_2')]

        graph = build_graph(external_graph_nodes, external_graph_edges)
        graph.stage = 'middle'
        partial_infer(graph)
        res_1 = Node(graph, 'res_1')
        res_2 = Node(graph, 'res_2')
        npt.assert_array_equal(res_1.in_port(0).data.get_shape(), int64_array([3]))
        npt.assert_array_equal(res_2.in_port(0).data.get_shape(), int64_array([3]))

    def test_fake_results(self):
        then_graph_nodes = {**valued_const_with_data('fake_const', int64_array(0)),
                            **regular_op_with_empty_data('shapeof',
                                         {'kind': 'op', 'type': 'ShapeOf', 'op': 'ShapeOf', 'infer': Shape.infer,
                                          'output_type': np.int64}),
                            **regular_op_with_empty_data('res_1', {'kind': 'op', 'type': 'Result', 'op': 'Result',
                                                                   'infer': lambda x: 0, 'output_id': 0})}
        then_graph_edges = [*connect('fake_const', 'shapeof'),
                            *connect('shapeof', 'res_1'),
                            ]

        else_graph_nodes = {**regular_op_with_empty_data('param_1', {'type': 'Parameter', 'kind': 'op', 'input_id': 1,
                                                                     'shape': None, 'infer': Parameter.infer}),
                            **regular_op_with_empty_data('res_1', {'kind': 'op', 'type': 'Result', 'op': 'Result',
                                                                   'infer': lambda x: 0, 'output_id': 0})}
        else_graph_edges = [*connect('param_1', 'res_1')]
        then_graph = build_graph_with_edge_attrs(then_graph_nodes, then_graph_edges)
        else_graph = build_graph_with_edge_attrs(else_graph_nodes, else_graph_edges)
        external_graph_nodes = {
            **valued_const_with_data('cond', np.array([True], dtype=np.bool)),
            **valued_const_with_data('input_1', int64_array([[1, 2, 3], [3, 2, 3]])),
            **regular_op_with_empty_data('if', {'kind': 'op', 'op': 'If', 'then_graph': then_graph,
                                                'else_graph': else_graph, 'infer': If.infer}),
            **result('res_1')}
        external_graph_edges = [*connect('cond', '0:if'),
                                *connect('input_1', '1:if'),
                                *connect('if', 'res_1')]

        graph = build_graph(external_graph_nodes, external_graph_edges)
        graph.stage = 'middle'
        partial_infer(graph)
        res_1 = Node(graph, 'res_1')
        npt.assert_array_equal(res_1.in_port(0).data.get_shape(), int64_array([2,3]))
