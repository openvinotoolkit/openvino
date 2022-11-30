# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.common.partial_infer.utils import int64_array, shape_array
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.middle.permute_tensor_iterator import TransposeTensorIteratorLSTM
from openvino.tools.mo.ops.parameter import Parameter
from openvino.tools.mo.ops.squeeze import Squeeze
from openvino.tools.mo.ops.unsqueeze import Unsqueeze
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, connect, connect_data, regular_op_with_shaped_data, \
    valued_const_with_data, regular_op_with_empty_data, result, shaped_const_with_data, empty_data


class PermuteTensorIteratorTest(unittest.TestCase):
    def test_permute_ti(self):
        nodes = {
            **regular_op_with_shaped_data('input', [3, 1, 10],
                                          {'type': 'Parameter', 'op': 'Parameter'}),
            **valued_const_with_data('direct_permute_order', int64_array([1, 0, 2])),
            **regular_op_with_empty_data('direct_permute', {'type': 'Transpose', 'op': 'Transpose'}),
            **regular_op_with_empty_data('init_hidden', {'op': 'SomeOp'}),
            **regular_op_with_empty_data('init_cell', {'op': 'SomeOp'}),
            **regular_op_with_shaped_data('ti', [3, 1, 10],
                                          {'type': 'TensorIterator', 'op': 'TensorIterator', 'body': None,
                                           'output_port_map': [{'axis': 1, 'start': -1, 'end': 0, 'stride': -1,
                                                                'external_port_id': 0}],
                                           'input_port_map': [{'axis': 1, 'start': 0, 'end': -1, 'stride': 1,
                                                               'external_port_id': 0}]}),
            **valued_const_with_data('inverse_permute_order', int64_array([1, 0, 2])),
            **regular_op_with_shaped_data('inverse_permute', [3, 1, 10],
                                          {'type': 'Transpose', 'op': 'Transpose', 'shape': [3, 1, 10]}),
            **result()
        }

        graph = build_graph(nodes,
                            [*connect('input', '0:direct_permute'),
                             *connect('direct_permute_order', '1:direct_permute'),
                             ('direct_permute', 'direct_permute_d'),
                             ('direct_permute_d', 'ti', {'in': 0, 'external_port_id': 0}),
                             *connect('init_hidden', '1:ti'),
                             *connect('init_cell', '2:ti'),
                             ('ti', 'ti_d', {'external_port_id': 0}),
                             ('ti_d', 'inverse_permute', {'in': 0}),
                             *connect('inverse_permute_order', '1:inverse_permute'),
                             *connect('inverse_permute', 'output')],
                            nodes_with_edges_only=True)

        nodes_internal = {
            **regular_op_with_shaped_data('input_unsqueezed', [3, 1, 10],
                                          {'type': 'Parameter', 'op': 'Parameter', 'shape': shape_array([3, 1, 10]),
                                           'infer': Parameter.infer}),
            **regular_op_with_empty_data('input_hidden', {'type': 'Parameter', 'op': 'Parameter'}),
            **regular_op_with_empty_data('input_cell', {'type': 'Parameter', 'op': 'Parameter'}),
            **valued_const_with_data("squeeze_dim", int64_array([1])),
            **valued_const_with_data("unsqueeze_dim", int64_array([1])),
            **regular_op_with_shaped_data('squeeze', [3, 10], {'type': 'Squeeze', 'op': 'Squeeze',
                                                               'infer': Squeeze.infer}),
            **regular_op_with_shaped_data('unsqueeze', [3, 1, 10],
                                          {'type': 'Unsqueeze', 'op': 'Unsqueeze',
                                           'infer': Unsqueeze.infer}),
            **shaped_const_with_data("weights", [1, 2, 3]),
            **shaped_const_with_data("biases", [1, 2, 3]),
            **regular_op_with_shaped_data('lstm', [3, 10], {'type': 'LSTMCell', 'op': 'LSTMCell', 'shape': [3, 10]}),
            **empty_data('lstm_d2'),
            **result('op_output'),
            **result('op_output_1'),
            **result('op_output_2')
        }

        sub_graph = build_graph(nodes_internal,
                                [*connect('input_unsqueezed', '0:squeeze'),
                                 *connect('squeeze_dim', '1:squeeze'),
                                 *connect('squeeze', '0:lstm'),
                                 *connect('input_hidden', '1:lstm'),
                                 *connect('input_cell', '2:lstm'),
                                 *connect('weights', '3:lstm'),
                                 *connect('biases', '4:lstm'),
                                 *connect('lstm:0', 'unsqueeze'),
                                 ('lstm', 'lstm_d2', {'out': 1}),
                                 *connect_data('lstm:0', 'op_output_1'),
                                 ('lstm_d2', 'op_output_2'),
                                 *connect('unsqueeze_dim', '1:unsqueeze'),
                                 *connect('unsqueeze', 'op_output')],
                                nodes_with_edges_only=True)

        ti_node = Node(graph, 'ti')
        ti_node.body = sub_graph

        TransposeTensorIteratorLSTM().find_and_replace_pattern(graph)
        graph.clean_up()

        graph_ref = build_graph(nodes,
                                [('input', 'input_d'),
                                 ('input_d', 'ti', {'in': 0, 'external_port_id': 0}),
                                 *connect('init_hidden', '1:ti'),
                                 *connect('init_cell', '2:ti'),
                                 ('ti', 'ti_d', {'external_port_id': 0}),
                                 ('ti_d', 'output', {'in': 0}), ],
                                {'input_d': {'shape': [3, 1, 10]},
                                 'ti': {'output_port_map': [{'axis': 0, 'start': -1, 'end': 0, 'stride': -1,
                                                             'external_port_id': 0}],
                                        'input_port_map': [{'axis': 0, 'start': 0, 'end': -1, 'stride': 1,
                                                            'external_port_id': 0}]},
                                 'ti_d': {'shape': [1, 3, 10]}}, nodes_with_edges_only=True)

        sub_graph_ref = build_graph(nodes_internal,
                                    [*connect('input_unsqueezed', '0:squeeze'),
                                     *connect('squeeze_dim', '1:squeeze'),
                                     *connect('squeeze', '0:lstm'),
                                     *connect('input_hidden', '1:lstm'),
                                     *connect('input_cell', '2:lstm'),
                                     *connect('weights', '3:lstm'),
                                     *connect('biases', '4:lstm'),
                                     *connect('lstm:0', 'unsqueeze'),
                                     ('lstm', 'lstm_d2', {'out': 1}),
                                     *connect_data('lstm:0', 'op_output_1'),
                                     ('lstm_d2', 'op_output_2'),
                                     *connect('unsqueeze_dim', '1:unsqueeze'),
                                     *connect('unsqueeze', 'op_output')],
                                    {'input_unsqueezed': {'shape': shape_array([1, 3, 10])},
                                     'input_unsqueezed_d': {'shape': [1, 3, 10]},
                                     'squeeze_dim': {'value': int64_array([0])},
                                     'unsqueeze_dim': {'value': int64_array([1])},
                                     }, nodes_with_edges_only=True)

        flag, resp = compare_graphs(
            sub_graph, sub_graph_ref, 'op_output', check_op_attrs=True)
        self.assertTrue(flag, resp)

        ti_node = Node(graph, 'ti')
        ti_node.body = None  # compare main graph without body
        flag, resp = compare_graphs(
            graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
