# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
from generator import generator, generate

from mo.graph.graph import Node
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.ir_reader.layer_to_class import groupconv_to_conv, restore_tensor_names
from unit_tests.utils.graph import build_graph


@generator
class TestFunction(unittest.TestCase):

    @generate(*[([1, 32, 112, 112], [32, 1, 1, 3], [32, 1, 1, 1, 3], 32),
                ([1, 32, 112, 112], [32, 1, 1, 1, 3], None, 32),
                ])
    def test_groupconv_to_conv(self, shape, weights_shape, reshape_shape, group):

        weights_const = np.random.randn(*weights_shape).astype(np.float32)

        nodes_attributes = {
            'input': {'kind': 'op', 'type': 'Parameter'},
            'input_data': {'shape': shape, 'kind': 'data'},

            'group_conv': {'kind': 'op', 'type': 'GroupConvolution'},
            'group_conv_data': {'shape': shape, 'kind': 'data'},

            'conv': {'kind': 'op', 'type': 'Convolution', 'group': group},
            'conv_data': {'shape': shape, 'kind': 'data'},

            'weights': {'kind': 'op', 'type': 'Const', 'value': weights_const},
            'weights_data': {'shape': weights_shape, 'kind': 'data'},

            'reshape': {'kind': 'op', 'type': 'Reshape'},
            'reshape_data': {'shape': reshape_shape, 'kind': 'data'},
            'reshape_const': {'kind': 'op', 'type': 'Const'},
            'reshape_const_data': {'shape': len(reshape_shape) if reshape_shape is not None else None, 'kind': 'data'},

            'add': {'kind': 'op', 'type': 'Add'},
            'add_data': {'shape': shape, 'kind': 'data'},
            'add_const': {'kind': 'op', 'type': 'Const'},
            'add_const_data': {'shape': [1, 32, 1, 1], 'kind': 'data'},
            'result': {'kind': 'op', 'type': 'Result'}
        }

        edges = [('input', 'input_data'),
                 ('input_data', 'group_conv'),
                 ('weights', 'weights_data'),
                 ('group_conv', 'group_conv_data'),
                 ('group_conv_data', 'add'),
                 ('add_const', 'add_const_data'),
                 ('add_const_data', 'add'),
                 ('add', 'add_data'),
                 ('add_data', 'result'),
                 ]

        if reshape_shape is not None:

            edges += [('weights_data', 'reshape'),
                      ('reshape_const', 'reshape_const_data'),
                      ('reshape_const_data', 'reshape'),
                      ('reshape', 'reshape_data'),
                      ('reshape_data', 'group_conv')]
        else:
            edges.append(('weights_data', 'group_conv'))

        graph = build_graph(nodes_attributes, edges, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('input', 'input_data'),
                                 ('input_data', 'conv'),
                                 ('weights', 'weights_data'),
                                 ('weights_data', 'conv'),
                                 ('conv', 'conv_data'),
                                 ('conv_data', 'add'),
                                 ('add_const', 'add_const_data'),
                                 ('add_const_data', 'add'),
                                 ('add', 'add_data'),
                                 ('add_data', 'result'),
                                 ], nodes_with_edges_only=True)

        for op in graph.get_op_nodes(type='GroupConvolution'):
            groupconv_to_conv(op)

        if reshape_shape is None:
            new_shape = [weights_shape[1] * group, *weights_shape[2:]]
            weights_const = np.reshape(weights_const, new_shape)
            node = Node(graph_ref, 'weights')
            node.value = weights_const

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_restore_tensor_names(self):

        shape = [1, 3, 224, 224]

        nodes_attributes = {
            'input': {'kind': 'op', 'type': 'Parameter', 'ports': {0: (shape, 'abc,def')}},
            'input_data': {'shape': shape, 'kind': 'data'},
            'add': {'kind': 'op', 'type': 'Add', 'ports': {2: (shape, r'ghi\,jkl')}},
            'add_data': {'shape': shape, 'kind': 'data'},
            'add_const': {'kind': 'op', 'type': 'Const', 'ports': {0: (shape, r'mno,pqr\,stu')}},
            'add_const_data': {'shape': shape, 'kind': 'data'},
            'result': {'kind': 'op', 'type': 'Result', 'ports': {0: (shape, None)}}
        }

        edges = [('input', 'input_data'),
                 ('input_data', 'add'),
                 ('add_const', 'add_const_data'),
                 ('add_const_data', 'add'),
                 ('add', 'add_data'),
                 ('add_data', 'result'),
                 ]

        graph = build_graph(nodes_attributes, edges, nodes_with_edges_only=True)

        for op in graph.get_op_nodes():
            restore_tensor_names(op)

        node_1 = Node(graph, 'input_data')
        node_2 = Node(graph, 'add_data')
        node_3 = Node(graph, 'add_const_data')

        assert node_1['fw_tensor_debug_info'] == [('abc', 'abc'), ('def', 'def')], 'Restored debug info is wrong!'
        assert node_2['fw_tensor_debug_info'] == [('ghi,jkl', 'ghi,jkl')], 'Restored debug info is wrong!'
        assert node_3['fw_tensor_debug_info'] == [('mno', 'mno'), ('pqr,stu', 'pqr,stu')],\
            'Restored debug info is wrong!'
