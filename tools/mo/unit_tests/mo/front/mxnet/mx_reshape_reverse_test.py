# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.mxnet.mx_reshape_reverse import MXReshapeReverse
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from unit_tests.utils.graph import build_graph


class TestMXReshapeReverseTest(unittest.TestCase):
    nodes_attributes = {
        'node_1': {'shape': int64_array([1, 2, 3, 4]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},

        'shape_node': {'kind': 'op', 'op': 'ShapeOf', 'type': 'ShapeOf'},
        'forward_reverse_unsqueeze_dims_node': {'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([0]), 'shape': int64_array([1])},
        'forward_reverse_unsqueeze_node': {'kind': 'op', 'op': 'Unsqueeze', 'type': 'Unsqueeze'},
        'forward_reverse_node': {'kind': 'op', 'op': 'Reverse', 'type': 'Reverse'},
        'forward_reverse_squeeze_dims_node': {'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([0]), 'shape': int64_array([1])},
        'forward_reverse_squeeze_node': {'kind': 'op', 'op': 'Squeeze', 'type': 'Squeeze'},
        'reshape_node': {'kind': 'op', 'op': 'Reshape', 'type': 'Reshape'},
        'reshape_shape_dim_node': {'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([3,2,1]), 'shape': int64_array([3])},
        'reshape_shape_node': {'kind': 'op', 'op': 'Reshape', 'type': 'Reshape'},
        'backward_shape_node': {'kind': 'op', 'op': 'ShapeOf', 'type': 'ShapeOf'},
        'backward_reverse_unsqueeze_dims_node': {'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([0]), 'shape': int64_array([1])},
        'backward_reverse_unsqueeze_node': {'kind': 'op', 'op': 'Unsqueeze', 'type': 'Unsqueeze'},
        'backward_reverse_node': {'kind': 'op', 'op': 'Reverse', 'type': 'Reverse'},
        'backward_reverse_squeeze_dims_node': {'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([0]), 'shape': int64_array([1])},
        'backward_reverse_squeeze_node': {'kind': 'op', 'op': 'Squeeze', 'type': 'Squeeze'},
        'last_reshape_node': {'kind': 'op', 'op': 'Reshape', 'type': 'Reshape'},
        'last': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
    }

    def test_mx_reshape_reverse(self):
        graph = build_graph({'node_1': {'shape': int64_array([1, 2, 3, 4]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                             'reshape': {'kind': 'op', 'op': 'MXReshape', 'dim': int64_array([1,2,3]), 'reverse': True},
                             'last': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
                             },
                            [('node_1', 'reshape', {'in': 0}),
                             ('reshape', 'last', {'in': 0}), ],
                             nodes_with_edges_only=True
                            )

        graph.stage = 'front'
        pattern = MXReshapeReverse()
        pattern.find_and_replace_pattern(graph)
        graph.clean_up()

        graph_ref = build_graph(self.nodes_attributes,
                                [('node_1', 'shape_node', {'in': 0, 'out': 0}),
                                 ('node_1', 'reshape_node', {'in': 0, 'out': 0}),
                                 ('shape_node', 'forward_reverse_unsqueeze_node', {'in': 0, 'out': 0}),
                                 ('forward_reverse_unsqueeze_dims_node', 'forward_reverse_unsqueeze_node', {'in': 1, 'out': 0}),
                                 ('forward_reverse_unsqueeze_node', 'forward_reverse_node', {'in': 0, 'out': 0}),
                                 ('forward_reverse_node', 'forward_reverse_squeeze_node', {'in': 0, 'out': 0}),
                                 ('forward_reverse_squeeze_dims_node', 'forward_reverse_squeeze_node', {'in': 1, 'out': 0}),
                                 ('forward_reverse_squeeze_node', 'reshape_node', {'in': 1, 'out': 0}),
                                 ('reshape_node', 'reshape_shape_node', {'in': 0, 'out': 0}),
                                 ('reshape_shape_dim_node', 'reshape_shape_node', {'in': 1, 'out': 0}),

                                 ('reshape_shape_node', 'backward_shape_node', {'in': 0, 'out': 0}),
                                 ('backward_shape_node', 'backward_reverse_unsqueeze_node', {'in': 0, 'out': 0}),
                                 ('backward_reverse_unsqueeze_dims_node', 'backward_reverse_unsqueeze_node', {'in': 1, 'out': 0}),
                                 ('backward_reverse_unsqueeze_node', 'backward_reverse_node', {'in': 0, 'out': 0}),
                                 ('backward_reverse_node', 'backward_reverse_squeeze_node', {'in': 0, 'out': 0}),
                                 ('backward_reverse_squeeze_dims_node', 'backward_reverse_squeeze_node', {'in': 1, 'out': 0}),

                                 ('backward_reverse_squeeze_node', 'last_reshape_node', {'in': 1, 'out': 0}),
                                 ('reshape_shape_node', 'last_reshape_node', {'in': 0, 'out': 0}),
                                 ('last_reshape_node', 'last', {'in': 0, 'out': 0}),
                                 ])
        graph_ref.clean_up()

        #Cannot use compare_graphs func. The outputs for some nodes not sorted.

        ref_nodes = graph_ref.get_op_nodes()
        nodes = graph.get_op_nodes()
        self.assertTrue(len(nodes) == len(ref_nodes))
        shapeof_count = 0
        ref_shapeof_count = 0
        reshape_count = 0
        ref_reshape_count = 0
        reverse_count = 0
        ref_reverse_count = 0

        for rnode in ref_nodes:
            if rnode['name'] == 'last':
                last_ref_node = rnode
            if rnode['op'] == 'ShapeOf':
                ref_shapeof_count = ref_shapeof_count + 1
            if rnode['op'] == 'Reshape':
                ref_reshape_count = ref_reshape_count + 1
            if rnode['op'] == 'Reverse':
                ref_reverse_count = ref_reverse_count + 1

        for node in nodes:
            if node['name'] == 'last':
                last_node = node
            if node['op'] == 'ShapeOf':
                shapeof_count = shapeof_count + 1
            if node['op'] == 'Reshape':
                reshape_count = reshape_count + 1
            if node['op'] == 'Reverse':
                reverse_count = reverse_count + 1

        self.assertTrue(shapeof_count == ref_shapeof_count)
        self.assertTrue(reshape_count == ref_reshape_count)
        self.assertTrue(reverse_count == ref_reverse_count)
        self.assertTrue(last_ref_node.op == last_node.op)
