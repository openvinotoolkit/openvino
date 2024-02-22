# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.middle.GroupNorm import GroupNormToMVN
from openvino.tools.mo.front.common.partial_infer.utils import float_array, int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, result, connect, \
    regular_op_with_shaped_data, valued_const_with_data

shape = int64_array([1, 3, 5, 2])
nodes = {**regular_op_with_shaped_data('input', shape, {'type': 'Parameter', 'op': 'Parameter'}),
         **valued_const_with_data('gamma', float_array([0.5])),
         **valued_const_with_data('beta', float_array([0.5])),
         **regular_op_with_shaped_data('group_norm', shape,
                                       {'op': 'GroupNorm', 'name': 'group_norm', 'num_groups': 3, 'eps': 1e-9}),
         **result('result')
         }

edges = [*connect('input:0', '0:group_norm'),
         *connect('gamma', '1:group_norm'),
         *connect('beta', '2:group_norm'),
         *connect('group_norm:0', 'result'),
         ]

ref_nodes = {**regular_op_with_shaped_data('input', shape, {'type': 'Parameter', 'op': 'Parameter'}),
             **regular_op_with_shaped_data('shape1', int64_array([4]), {'op': 'ShapeOf'}),
             **regular_op_with_shaped_data('shape2', int64_array([4]), {'op': 'ShapeOf'}),
             **regular_op_with_shaped_data('shape3', int64_array([1]), {'op': 'ShapeOf'}),
             **regular_op_with_shaped_data('hcast1', int64_array([4]), {'op': 'Cast'}),
             **regular_op_with_shaped_data('cast2', int64_array([2]), {'op': 'Cast'}),
             **regular_op_with_shaped_data('cast3', int64_array([4]), {'op': 'Cast'}),
             **regular_op_with_shaped_data('gather1', int64_array([2]), {'op': 'Gather'}),
             **regular_op_with_shaped_data('gather2', int64_array([1]), {'op': 'Gather'}),
             **regular_op_with_shaped_data('gather3', int64_array([1]), {'op': 'Gather'}),
             **regular_op_with_shaped_data('mul1', int64_array([1]), {'op': 'Mul'}),
             **regular_op_with_shaped_data('mul2', int64_array([1]), {'op': 'Mul'}),
             **regular_op_with_shaped_data('mul3', shape, {'op': 'Mul'}),
             **regular_op_with_shaped_data('concat', int64_array([4]), {'op': 'Concat'}),
             **regular_op_with_shaped_data('reshape1', int64_array([3, 1, 5, 2]), {'op': 'Reshape'}),
             **regular_op_with_shaped_data('reshape2', shape, {'op': 'Reshape'}),
             **regular_op_with_shaped_data('squeeze', int64_array([]), {'op': 'Squeeze'}),
             **regular_op_with_shaped_data('range', int64_array([3]), {'op': 'Range'}),
             **regular_op_with_shaped_data('mvn', int64_array([3, 1, 5, 2]), {'op': 'MVN'}),
             **regular_op_with_shaped_data('add', shape, {'op': 'Add'}),
             **valued_const_with_data('shape/axis1', int64_array(0)),
             **valued_const_with_data('shape/ind1', int64_array([2, 3])),
             **valued_const_with_data('shape/axis2', int64_array(0)),
             **valued_const_with_data('shape/ind2', int64_array([0])),
             **valued_const_with_data('shape/axis3', int64_array(0)),
             **valued_const_with_data('shape/ind3', int64_array([1])),
             **valued_const_with_data('gn/rec', float_array([1./3])),
             **valued_const_with_data('group', int64_array([3])),
             **valued_const_with_data('squeeze/axis', int64_array([0])),
             **valued_const_with_data('range/start', int64_array(1)),
             **valued_const_with_data('range/step', int64_array(1)),
             **valued_const_with_data('gamma', float_array([[[[0.5]]]])),
             **valued_const_with_data('beta', float_array([[[[0.5]]]])),
             **result('result')
             }
ref_edges = [*connect('input', '0:reshape1'),
             *connect('input', 'shape1', skip_data=True),
             *connect('shape1:0', '0:gather1'),
             *connect('shape1:0', 'hcast1', skip_data=True),
             *connect('shape/ind1', '1:gather1'),
             *connect('shape/axis1', '2:gather1'),
             *connect('gather1', 'cast2'),
             *connect('hcast1', '0:gather3'),
             *connect('hcast1', '0:gather2', skip_data=True),
             *connect('shape/ind2', '1:gather2'),
             *connect('shape/axis2', '2:gather2'),
             *connect('gather2', '0:mul2'),
             *connect('group', '1:mul2'),
             *connect('shape/ind3', '1:gather3'),
             *connect('shape/axis3', '2:gather3'),
             *connect('gather3', '0:mul1'),
             *connect('gn/rec', '1:mul1'),
             *connect('mul2', '0:concat'),
             *connect('mul1', '1:concat'),
             *connect('cast2', '2:concat'),
             *connect('concat', 'cast3'),
             *connect('cast3', '1:reshape1'),
             *connect('reshape1', 'shape2'),
             *connect('shape2', 'shape3'),
             *connect('shape3', '0:squeeze'),
             *connect('squeeze/axis', '1:squeeze'),
             *connect('range/start', '0:range'),
             *connect('squeeze', '1:range'),
             *connect('range/step', '2:range'),
             *connect('reshape1', '0:mvn', skip_data=True),
             *connect('range', '1:mvn'),
             *connect('mvn', '0:reshape2'),
             *connect('shape1:0', '1:reshape2', skip_data=True),
             *connect('reshape2', '0:mul3'),
             *connect('gamma', '1:mul3'),
             *connect('mul3', '0:add'),
             *connect('beta', '1:add'),
             *connect('add', 'result')
             ]


class GroupNormToMVNTest(unittest.TestCase):
    def test_group_norm_1(self):
        graph = build_graph(nodes, edges)

        graph_ref = build_graph(ref_nodes, ref_edges)

        graph.graph['layout'] = 'NCHW'

        GroupNormToMVN().find_and_replace_pattern(graph)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)
