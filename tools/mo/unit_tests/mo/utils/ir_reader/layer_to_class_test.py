# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np

import openvino.tools.mo.graph.graph
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from openvino.tools.mo.utils.ir_reader.internal_ops.squeeze import SqueezeInternal
from openvino.tools.mo.utils.ir_reader.internal_ops.unsqueeze import UnsqueezeInternal
from openvino.tools.mo.utils.ir_reader.layer_to_class import groupconv_to_conv, restore_tensor_names
from unit_tests.utils.graph import build_graph
from unit_tests.utils.graph import connect_data,shaped_parameter, regular_op_with_shaped_data, connect, valued_const_with_data, shaped_const_with_data, result
from openvino.tools.mo.ops.op import Op


class TestFunction():
    @pytest.mark.parametrize("shape, weights_shape, reshape_shape, group",[([1, 32, 112, 112], [32, 1, 1, 3], [32, 1, 1, 1, 3], 32),
                ([1, 32, 112, 112], [32, 1, 1, 1, 3], None, 32),
                ])
    def test_groupconv_to_conv(self, shape, weights_shape, reshape_shape, group):
        weights_const = np.random.randn(*weights_shape).astype(np.float32)

        nodes_attributes = {
            **shaped_parameter('input', shape),
            **regular_op_with_shaped_data('group_conv', shape, {'type': 'GroupConvolution'}),
            **regular_op_with_shaped_data('conv', shape, {'type': 'Convolution'}),
            **valued_const_with_data('weights', weights_const, weights_shape, {'type': 'Const'}),
            **regular_op_with_shaped_data('reshape', reshape_shape, {'type': 'Reshape'}),
            **shaped_const_with_data('reshape_const', len(reshape_shape) if reshape_shape is not None else None, {'type': 'Const'}),
            **regular_op_with_shaped_data('add', shape, {'type': 'Add'}),
            **shaped_const_with_data('add_const', [1, 32, 1, 1], {'type': 'Const'}),
            **result("result")
        }

        edges = [*connect('input:0', 'group_conv:0'),
                 *connect('group_conv:0', 'add:0'),
                 *connect('add_const:0', 'add:1'),
                 *connect('add:0', 'result:0'),
                 ]

        if reshape_shape is not None:

            edges += [*connect('weights:0', 'reshape:0'),
                      *connect('reshape_const:0', 'reshape:1'),
                      *connect('reshape:0', 'group_conv:1')]
        else:
            edges += [*connect('weights:0', 'group_conv:1')]

        graph = build_graph(nodes_attributes, edges)
        reshape_node = None
        if reshape_shape is None:
            reshape_node = Node(graph, 'reshape')

        graph_ref = build_graph(nodes_attributes,
                                [*connect('input:0', 'conv:0'),
                                 *connect('weights:0', 'conv:1'),
                                 *connect('conv:0', 'add:0'),
                                 *connect('add_const:0', 'add:1'),
                                 *connect('add:0', 'result:0'),
                                 ])
        for op in graph.get_op_nodes(type='GroupConvolution'):
            groupconv_to_conv(op)

        if reshape_shape is None:
            new_shape = [weights_shape[1] * group, *weights_shape[2:]]
            weights_const = np.reshape(weights_const, new_shape)
            node = Node(graph_ref, 'weights')
            node.value = weights_const

            assert len(reshape_node.in_nodes()) == 0 and len(reshape_node.out_nodes()) == 0

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        assert flag, resp

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
        assert node_3['fw_tensor_debug_info'] == [('mno', 'mno'), ('pqr,stu', 'pqr,stu')], \
            'Restored debug info is wrong!'

    def test_squeeze(self):
        nodes_attributes = {
            'input': {'kind': 'op', 'type': 'Parameter'},
            'input_data': {'shape': [2, 1, 3], 'kind': 'data'},

            'axis': {'kind': 'op', 'type': 'Const', 'op': 'Const', 'value': np.array(1), 'shape': []},
            'axis_data': {'shape': [], 'kind': 'data', 'value': np.array(1)},

            'squeeze': {'kind': 'op', 'type': 'Squeeze'},
            'squeeze_data': {'shape': [2, 3], 'kind': 'data', 'value': None},

            'result': {'kind': 'op', 'type': 'Result'}
        }

        edges = [('input', 'input_data'),
                 ('input_data', 'squeeze'),
                 ('axis', 'axis_data'),
                 ('axis_data', 'squeeze'),
                 ('squeeze', 'squeeze_data'),
                 ('squeeze_data', 'result'),
                 ]

        graph = build_graph(nodes_attributes, edges, nodes_with_edges_only=True)

        squeeze_node = Node(graph, 'squeeze')
        SqueezeInternal.infer(squeeze_node)

        graph_ref = build_graph(nodes_attributes, edges, nodes_with_edges_only=True)

        # Check that graph wasn't changed after shape infer
        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        assert flag, resp

    def test_squeeze_no_axes(self):
        nodes_attributes = {
            'input': {'kind': 'op', 'type': 'Parameter'},
            'input_data': {'shape': [2, 1, 3], 'kind': 'data'},

            'squeeze': {'kind': 'op', 'type': 'Squeeze'},
            'squeeze_data': {'shape': [2, 3], 'kind': 'data', 'value': None},

            'result': {'kind': 'op', 'type': 'Result'}
        }

        edges = [('input', 'input_data'),
                 ('input_data', 'squeeze'),
                 ('squeeze', 'squeeze_data'),
                 ('squeeze_data', 'result'),
                 ]

        graph = build_graph(nodes_attributes, edges, nodes_with_edges_only=True)

        squeeze_node = Node(graph, 'squeeze')
        SqueezeInternal.infer(squeeze_node)

        graph_ref = build_graph(nodes_attributes, edges, nodes_with_edges_only=True)

        # Check that graph wasn't changed after shape infer
        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        assert flag, resp

    def test_unsqueeze(self):
        nodes_attributes = {
            'input': {'kind': 'op', 'type': 'Parameter'},
            'input_data': {'shape': [2, 3], 'kind': 'data'},

            'axis': {'kind': 'op', 'type': 'Const', 'op': 'Const', 'value': np.array(1), 'shape': []},
            'axis_data': {'shape': [], 'kind': 'data', 'value': np.array(1)},

            'unsqueeze': {'kind': 'op', 'type': 'Unsqueeze'},
            'unsqueeze_data': {'shape': [2, 1, 3], 'kind': 'data', 'value': None},

            'result': {'kind': 'op', 'type': 'Result'}
        }

        edges = [('input', 'input_data'),
                 ('input_data', 'unsqueeze'),
                 ('axis', 'axis_data'),
                 ('axis_data', 'unsqueeze'),
                 ('unsqueeze', 'unsqueeze_data'),
                 ('unsqueeze_data', 'result'),
                 ]

        graph = build_graph(nodes_attributes, edges, nodes_with_edges_only=True)

        unsqueeze_node = Node(graph, 'unsqueeze')
        UnsqueezeInternal.infer(unsqueeze_node)

        graph_ref = build_graph(nodes_attributes, edges, nodes_with_edges_only=True)

        # Check that graph wasn't changed after shape infer
        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        assert flag, resp
