# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.middle.passes.fusing.helpers import forward_bfs, backward_bfs, get_next_operation, common_bfs
from unit_tests.utils.graph import build_graph, regular_op_with_shaped_data, connect, result, \
    valued_const_with_data, connect_data

nodes_attributes = {
    'placeholder_1': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # ScaleShift layer
    'scaleshift_1': {'type': 'ScaleShift', 'kind': 'op', 'op': 'ScaleShift'},
    'scaleshift_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'scaleshift_1_b': {'value': None, 'shape': None, 'kind': 'data'},
    'scaleshift_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Mul and Add operations
    'mul_1': {'type': 'Mul', 'kind': 'op', 'op': 'Mul'},
    'mul_1_w': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'mul_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'add_1': {'type': 'Add', 'kind': 'op', 'op': 'Add'},
    'add_1_w': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'add_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # Mul2 and Add2 operations
    'mul_2': {'type': 'Mul', 'kind': 'op', 'op': 'Mul'},
    'mul_2_w': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'mul_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'add_2': {'type': 'Add', 'kind': 'op', 'op': 'Add'},
    'add_2_w': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'add_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # Concat1 operation
    'concat_1': {'type': 'Concat', 'kind': 'op', 'op': 'Concat'},
    'concat_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Convolutions
    'conv_1': {'type': 'Convolution', 'kind': 'op', 'op': 'Conv2D', 'layout': 'NHWC'},
    'conv_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_1_b': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_2': {'type': 'Convolution', 'kind': 'op', 'op': 'Conv2D', 'layout': 'NHWC'},
    'conv_2_w': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_2_b': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_2_data': {'value': None, 'shape': None, 'kind': 'data'},
    # FullyConnected
    'fc_1': {'type': 'MatMul', 'kind': 'op', 'op': 'FullyConnected', 'layout': 'NHWC'},
    'fc_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'fc_1_b': {'value': None, 'shape': None, 'kind': 'data'},
    'fc_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Placeholders
    'placeholder_2': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_3': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_3_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'op_output': { 'kind': 'op', 'op': 'Result'}
}


# Unit tests for forward and backward bfs (forward_bfs, backward_bfs)
class BFSTests(unittest.TestCase):
    def test_forward_bfs_simple(self):
        # Placeholder->ScaleShift->Mul->Add
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'scaleshift_1'),
                             ('scaleshift_1_w', 'scaleshift_1'),
                             ('scaleshift_1', 'scaleshift_1_data'),
                             ('scaleshift_1_data', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'op_output')
                             ])

        res = forward_bfs(Node(graph, 'placeholder_1'), ['ScaleShift', 'Mul'], ['Add'])
        self.assertTrue(len(res) == 1 and res[0].id == 'add_1', 'Add operation was not found by bfs')

        res = forward_bfs(Node(graph, 'placeholder_1'), [], ['Add'], allowed_all=True)
        self.assertTrue(len(res) == 1 and res[0].id == 'add_1', 'Add operation was not found by bfs')

        res = forward_bfs(Node(graph, 'placeholder_1_data'), ['ScaleShift'], ['Add'])
        self.assertTrue(len(res) == 0, 'No one node should be found! But bfs found {} nodes'.format(len(res)))

        res = forward_bfs(Node(graph, 'placeholder_1_data'), ['ScaleShift'], ['Mul', 'Add'])
        self.assertTrue(len(res) == 1 and res[0].id == 'mul_1', 'BFS should find only one Mul operation')

    def test_backward_bfs_simple(self):
        # Placeholder->ScaleShift->Mul->Add
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'scaleshift_1'),
                             ('scaleshift_1_w', 'scaleshift_1'),
                             ('scaleshift_1', 'scaleshift_1_data'),
                             ('scaleshift_1_data', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'op_output')
                             ])

        res = backward_bfs(Node(graph, 'add_1_data'), ['Add', 'ScaleShift', 'Mul'], ['Parameter'])
        self.assertTrue(len(res) == 1 and res[0].id == 'placeholder_1', 'Placeholder operation was not found by bfs')

        res = backward_bfs(Node(graph, 'add_1'), [], ['Parameter'], allowed_all=True)
        self.assertTrue(len(res) == 1 and res[0].id == 'placeholder_1', 'Placeholder operation was not found by bfs')

        res = backward_bfs(Node(graph, 'add_1_data'), ['Add'], ['ScaleShift'])
        self.assertTrue(len(res) == 0, 'No one node should be found! But bfs found {} nodes'.format(len(res)))

        res = backward_bfs(Node(graph, 'add_1_data'), ['Add', 'Mul'], ['Parameter', 'ScaleShift'])
        self.assertTrue(len(res) == 1 and res[0].id == 'scaleshift_1', 'BFS should find only one ScaleShift operation')

    def test_forward_bfs_hard(self):
        # Placeholder->ScaleShift->Mul1->Add1---->Concat
        #             `----------->Add2->Mul2--'
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'scaleshift_1'),
                             ('placeholder_1_data', 'add_2'),
                             ('scaleshift_1_w', 'scaleshift_1'),
                             ('scaleshift_1', 'scaleshift_1_data'),
                             ('scaleshift_1_data', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_2', 'add_2_data'),
                             ('add_2_data', 'mul_2'),
                             ('mul_2', 'mul_2_data'),
                             ('add_1_data', 'concat_1'),
                             ('mul_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('concat_1_data', 'op_output')
                             ])

        res = forward_bfs(Node(graph, 'placeholder_1'), ['ScaleShift', 'Mul', 'Add'], ['Concat'])
        self.assertTrue(len(res) == 1 and res[0].id == 'concat_1', 'Probably Concat operation was not found by bfs')

        res = forward_bfs(Node(graph, 'placeholder_1'), ['ScaleShift', 'Mul'], ['Add'])
        self.assertTrue(len(res) == 2 and all([res[x].id in ['add_1', 'add_2'] for x in range(len(res))]),
                        'Add operations was not found by bfs')

        res = forward_bfs(Node(graph, 'placeholder_1'), ['ScaleShift'], ['Add'])
        self.assertTrue(len(res) == 0, 'BFS shouldn\'t find any operations')

        res = forward_bfs(Node(graph, 'placeholder_1'), [], ['Add'], allowed_all=True)
        self.assertTrue(len(res) == 2 and all([res[x].id in ['add_1', 'add_2'] for x in range(len(res))]),
                        'Add operations was not found by bfs')

        res = forward_bfs(Node(graph, 'placeholder_1_data'), ['ScaleShift'], ['Concat'])
        self.assertTrue(len(res) == 0, 'No one node should be found! But bfs found {} nodes'.format(len(res)))

    def test_backward_bfs_hard(self):
        # Placeholder->ScaleShift->Mul1->Add1---->Concat
        #             `----------->Add2->Mul2--'
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'scaleshift_1'),
                             ('placeholder_1_data', 'add_2'),
                             ('scaleshift_1_w', 'scaleshift_1'),
                             ('scaleshift_1', 'scaleshift_1_data'),
                             ('scaleshift_1_data', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_2', 'add_2_data'),
                             ('add_2_data', 'mul_2'),
                             ('mul_2', 'mul_2_data'),
                             ('add_1_data', 'concat_1'),
                             ('mul_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('concat_1_data', 'op_output')
                             ])

        res = backward_bfs(Node(graph, 'concat_1'), ['ScaleShift', 'Mul', 'Add'], ['Parameter'])
        self.assertTrue(len(res) == 0, 'Smth went wrong with bfs')

        res = backward_bfs(Node(graph, 'concat_1'), ['Mul'], ['Add'])
        self.assertTrue(len(res) == 2 and all([res[x].id in ['add_1', 'add_2'] for x in range(len(res))]),
                        'Add operations was not found by bfs')

        res = backward_bfs(Node(graph, 'concat_1'), ['ScaleShift'], ['Add'])
        self.assertTrue(len(res) == 0, 'BFS shouldn\'t find any operations')

        res = backward_bfs(Node(graph, 'concat_1'), [], ['Add'], allowed_all=True)
        self.assertTrue(len(res) == 2 and all([res[x].id in ['add_1', 'add_2'] for x in range(len(res))]),
                        'Add operations was not found by bfs')

        res = backward_bfs(Node(graph, 'concat_1'), ['ScaleShift'], ['ScaleShift'])
        self.assertTrue(len(res) == 0, 'No one node should be found! But bfs found {} nodes'.format(len(res)))

    def test_backward_bfs_hard2(self):
        # Placeholder->ScaleShift->Mul1->Add1---->Concat
        #             `----------->Add2->Mul2--'
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'add_2'),
                             ('scaleshift_1_w', 'scaleshift_1'),
                             ('scaleshift_1', 'scaleshift_1_data'),
                             ('scaleshift_1_data', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_2', 'add_2_data'),
                             ('add_2_data', 'mul_2'),
                             ('mul_2', 'mul_2_data'),
                             ('add_1_data', 'concat_1'),
                             ('mul_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('concat_1_data', 'op_output')
                             ])

        res = backward_bfs(Node(graph, 'concat_1'), ['Mul', 'Add'], ['Parameter'])
        self.assertTrue(len(res) == 0, 'Smth went wrong with bfs')

        res = backward_bfs(Node(graph, 'concat_1'), ['Mul'], ['Add'])
        self.assertTrue(len(res) == 2 and all([res[x].id in ['add_1', 'add_2'] for x in range(len(res))]),
                        'Add operations was not found by bfs')

        res = backward_bfs(Node(graph, 'concat_1'), ['ScaleShift'], ['Add'])
        self.assertTrue(len(res) == 0, 'BFS shouldn\'t find any operations')

        res = backward_bfs(Node(graph, 'concat_1'), [], ['Add'], allowed_all=True)
        self.assertTrue(len(res) == 2 and all([res[x].id in ['add_1', 'add_2'] for x in range(len(res))]),
                        'Add operations was not found by bfs')

        res = backward_bfs(Node(graph, 'concat_1'), ['ScaleShift'], ['ScaleShift'])
        self.assertTrue(len(res) == 0, 'No one node should be found! But bfs found {} nodes'.format(len(res)))

    def test_backward_bfs_cycle(self):
        # Placeholder->ScaleShift->Mul->Add
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'scaleshift_1'),
                             ('scaleshift_1_w', 'scaleshift_1'),
                             ('scaleshift_1', 'scaleshift_1_data'),
                             ('scaleshift_1_data', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'placeholder_1'),
                             ('add_1_data', 'op_output')
                             ])

        res = backward_bfs(Node(graph, 'add_1_data'), ['Add', 'ScaleShift', 'Mul', 'Parameter'], ['Conv2D'])
        self.assertTrue(len(res) == 0, 'Sholdn\'t find any nodes due to cycle in graph')

    def test_backward_bfs_check_op_instead_of_type(self):
        # Placeholder->ScaleShift->Mul1->Add1---->Concat
        #             `----------->Add2->Mul2--'
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'add_2'),
                             ('scaleshift_1_w', 'scaleshift_1'),
                             ('scaleshift_1', 'scaleshift_1_data'),
                             ('scaleshift_1_data', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_2', 'add_2_data'),
                             ('add_2_data', 'mul_2'),
                             ('mul_2', 'mul_2_data'),
                             ('add_1_data', 'concat_1'),
                             ('mul_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('concat_1_data', 'op_output')
                             ])

        res = common_bfs(Node(graph, 'concat_1'), ['Mul', 'Add'], ['Parameter'], is_backward=True, attr_to_check='op')
        self.assertTrue(len(res) == 0, 'Smth went wrong with bfs')

        res = common_bfs(Node(graph, 'concat_1'), ['Mul'], ['Add'], is_backward=True, attr_to_check='op')
        self.assertTrue(len(res) == 2 and all([res[x].id in ['add_1', 'add_2'] for x in range(len(res))]),
                        'Add operations was not found by bfs')

        res = common_bfs(Node(graph, 'concat_1'), ['ScaleShift'], ['Add'], is_backward=True, attr_to_check='op')
        self.assertTrue(len(res) == 0, 'BFS shouldn\'t find any operations')

        res = common_bfs(Node(graph, 'concat_1'), [], ['Add'], allowed_all=True, is_backward=True, attr_to_check='op')
        self.assertTrue(len(res) == 2 and all([res[x].id in ['add_1', 'add_2'] for x in range(len(res))]),
                        'Add operations was not found by bfs')

        res = common_bfs(Node(graph, 'concat_1'), ['ScaleShift'], ['ScaleShift'], is_backward=True, attr_to_check='op')
        self.assertTrue(len(res) == 0, 'No one node should be found! But bfs found {} nodes'.format(len(res)))

    def test_backward_bfs_multi_consumer_data_nodes(self):
        # Placeholder-> Mul -> Result
        # Const      -/    \- Result2

        graph = build_graph({**regular_op_with_shaped_data('parameter', [1], {'op': 'Parameter'}),
                             **valued_const_with_data('const', int64_array([5])),
                             **regular_op_with_shaped_data('mul', [1], {'op': 'Mul'}),
                             **result('result'),
                             **result('result2'),
                             },
                            [*connect('parameter', '0:mul'),
                             *connect('const', '1:mul'),
                             *connect('mul:0', 'result'),
                             *connect_data('mul', 'result2'),
                             ])

        res = common_bfs(Node(graph, 'result'), ['Mul'], ['Parameter'], is_backward=True, attr_to_check='op',
                         follow_multi_consumer_data_nodes=True)
        self.assertTrue(len(res) == 1, 'The multi-consumer data node "mul_d" was not followed')

        res = common_bfs(Node(graph, 'result'), ['Mul'], ['Parameter'], is_backward=True, attr_to_check='op')
        self.assertTrue(len(res) == 0, 'The multi-consumer data node "mul_d" was followed')


# Unit tests for get_next_operation
class GetNextOperationTests(unittest.TestCase):
    def test_get_next_operation_1(self):
        # Placeholder->ScaleShift->Mul->Add
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'scaleshift_1'),
                             ('scaleshift_1_w', 'scaleshift_1'),
                             ('scaleshift_1', 'scaleshift_1_data'),
                             ('scaleshift_1_data', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'op_output')
                             ])

        res = get_next_operation(Node(graph, 'mul_1'))
        self.assertTrue(len(res) == 1 and res[0].id == 'add_1', 'get_nex_operation returned wrong op')

    def test_get_next_operation_2(self):
        # Placeholder->ScaleShift->Mul->Add
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('placeholder_1_data', 'add_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'op_output')
                             ])

        res = get_next_operation(Node(graph, 'placeholder_1'))
        self.assertTrue(len(res) == 2 and all([x.id in ['add_1', 'mul_1'] for x in res]),
                        'get_nex_operation returned wrong op')

    def test_get_next_operation_3(self):
        # Placeholder-+--->ScaleShift
        #             +-----^
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1', 'placeholder_2_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('placeholder_2_data', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'op_output')
                             ])

        res = get_next_operation(Node(graph, 'placeholder_1'))
        self.assertTrue(len(res) == 1 and res[0].id == 'mul_1', 'get_nex_operation returned wrong op')
