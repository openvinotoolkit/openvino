"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import unittest
from argparse import Namespace

from extensions.back.WeightsQuantize import WeightsQuantize
from mo.utils.unittest.graph import build_graph, compare_graphs

nodes_attributes = {
    # placeholder
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {'kind': 'data'},

    # weights
    'weights_const': {'type': 'Const', 'kind': 'op'},
    'weights_data': {'kind': 'data'},

    # quantize
    'quantize1': {'type': 'FakeQuantize', 'kind': 'op', 'levels': 5, 'keep_in_IR': True},
    'quantize2': {'type': 'FakeQuantize', 'kind': 'op', 'levels': 2, 'keep_in_IR': True},
    'quantize3': {'type': 'FakeQuantize', 'kind': 'op', 'levels': None, 'keep_in_IR': True},
    'quantize4': {'type': 'FakeQuantize', 'kind': 'op', 'levels': 122, 'keep_in_IR': True},
    'quantize5': {'type': 'FakeQuantize', 'kind': 'op', 'levels': 202, 'keep_in_IR': True},
    'quantize6': {'type': 'FakeQuantize', 'kind': 'op', 'levels': 257, 'keep_in_IR': True},
    'quantize_data': {'kind': 'data'},
    'new_quantize1': {'kind': 'op', 'type': 'FakeQuantize', 'levels': 5},
    'new_quantize4': {'kind': 'op', 'type': 'FakeQuantize', 'levels': 122},
    'new_quantize5': {'kind': 'op', 'type': 'FakeQuantize', 'levels': 202},
    'new_quantize_data': {'kind': 'data'},

    #  quantize input/output
    'output_high_init': {'kind': 'op', 'type': 'Const'},
    'output_high_init_data': {'kind': 'data', 'value': 3},
    'output_low_init': {'kind': 'op', 'type': 'Const'},
    'output_low_init_data': {'kind': 'data', 'value': -1.5},

    'input_low': {'kind': 'op', 'type': 'Const'},
    'input_low_data': {'kind': 'data'},
    'input_high': {'kind': 'op', 'type': 'Const'},
    'input_high_data': {'kind': 'data'},

    'output_low': {'kind': 'op', 'type': 'Const'},
    'output_low_data': {'kind': 'data'},
    'output_high': {'kind': 'op', 'type': 'Const'},
    'output_high_data': {'kind': 'data'},

    'output_high_init_data1': {'kind': 'data', 'value': 256.1},
    'output_low_init_data1': {'kind': 'data', 'value': 17.3},

    'output_high_init_data2': {'kind': 'data', 'value': -0.42},
    'output_low_init_data2': {'kind': 'data', 'value': -2.573},

    # eltwise ops
    'mul': {'kind': 'op', 'op': 'Mul'},
    'add': {'kind': 'op', 'op': 'Add'},

    'scale': {'kind': 'op', 'type': 'Const', 'value': 1.125},
    'shift': {'kind': 'op', 'type': 'Const', 'value': -1.5},
    'scale1': {'kind': 'op', 'type': 'Const', 'value': 1.9735537190082646},
    'shift1': {'kind': 'op', 'type': 'Const', 'value': 17.3},
    'scale2': {'kind': 'op', 'type': 'Const', 'value': 0.010711442786069652},
    'shift2': {'kind': 'op', 'type': 'Const', 'value': -2.573},

    'shift_data': {'kind': 'data'},
    'scale_data': {'kind': 'data'},
    'mul_data': {'kind': 'data'},
    'add_data': {'kind': 'data'},

    'convolution': {'type': 'Convolution', 'kind': 'op'}
}


class WeightQuantizeTest(unittest.TestCase):

    def test_negative_quantize(self):
        graph = build_graph(nodes_attributes,
                            [('weights_const', 'weights_data'),
                             ('weights_data', 'mul'),
                             ('scale', 'mul'),
                             ('mul', 'add'),
                             ('shift', 'add'),
                             ('add', 'quantize_data'),
                             ('quantize_data', 'convolution')],
                            nodes_with_edges_only=True)
        graph.graph['cmd_params'] = Namespace(keep_quantize_ops_in_IR=True)

        graph_ref = build_graph(nodes_attributes,
                                [('weights_const', 'weights_data'),
                                 ('weights_data', 'mul'),
                                 ('scale', 'mul'),
                                 ('mul', 'add'),
                                 ('shift', 'add'),
                                 ('add', 'quantize_data'),
                                 ('quantize_data', 'convolution')],
                                nodes_with_edges_only=True)

        WeightsQuantize().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'convolution', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_negative_quantize_levels_2(self):
        graph = build_graph(nodes_attributes,
                            [('weights_const', 'weights_data'),
                             ('weights_data', 'quantize2', {'in': 0}),
                             ('input_low', 'input_low_data'),
                             ('input_low_data', 'quantize2', {'in': 1}),
                             ('input_high', 'input_high_data'),
                             ('input_high_data', 'quantize2', {'in': 2}),
                             ('output_low_init', 'output_low_init_data'),
                             ('output_low_init_data', 'quantize2', {'in': 3}),
                             ('output_high_init', 'output_high_init_data'),
                             ('output_high_init_data', 'quantize2', {'in': 4}),
                             ('quantize2', 'quantize_data'),
                             ('quantize_data', 'convolution', {'in': 1})],
                            nodes_with_edges_only=True)

        graph.graph['cmd_params'] = Namespace(keep_quantize_ops_in_IR=True)

        graph_ref = build_graph(nodes_attributes,
                                [('weights_const', 'weights_data'),
                                 ('weights_data', 'quantize2', {'in': 0}),
                                 ('input_low', 'input_low_data'),
                                 ('input_low_data', 'quantize2', {'in': 1}),
                                 ('input_high', 'input_high_data'),
                                 ('input_high_data', 'quantize2', {'in': 2}),
                                 ('output_low_init', 'output_low_init_data'),
                                 ('output_low_init_data', 'quantize2', {'in': 3}),
                                 ('output_high_init', 'output_high_init_data'),
                                 ('output_high_init_data', 'quantize2', {'in': 4}),
                                 ('quantize2', 'quantize_data'),
                                 ('quantize_data', 'convolution', {'in': 1})],
                                nodes_with_edges_only=True)

        WeightsQuantize().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'convolution', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_negative_quantize_levels_257(self):
        graph = build_graph(nodes_attributes,
                            [('weights_const', 'weights_data'),
                             ('weights_data', 'quantize6', {'in': 0}),
                             ('input_low', 'input_low_data'),
                             ('input_low_data', 'quantize6', {'in': 1}),
                             ('input_high', 'input_high_data'),
                             ('input_high_data', 'quantize6', {'in': 2}),
                             ('output_low_init', 'output_low_init_data'),
                             ('output_low_init_data', 'quantize6', {'in': 3}),
                             ('output_high_init', 'output_high_init_data'),
                             ('output_high_init_data', 'quantize6', {'in': 4}),
                             ('quantize6', 'quantize_data'),
                             ('quantize_data', 'convolution', {'in': 1})],
                            nodes_with_edges_only=True)

        graph.graph['cmd_params'] = Namespace(keep_quantize_ops_in_IR=True)

        graph_ref = build_graph(nodes_attributes,
                                [('weights_const', 'weights_data'),
                                 ('weights_data', 'quantize6', {'in': 0}),
                                 ('input_low', 'input_low_data'),
                                 ('input_low_data', 'quantize6', {'in': 1}),
                                 ('input_high', 'input_high_data'),
                                 ('input_high_data', 'quantize6', {'in': 2}),
                                 ('output_low_init', 'output_low_init_data'),
                                 ('output_low_init_data', 'quantize6', {'in': 3}),
                                 ('output_high_init', 'output_high_init_data'),
                                 ('output_high_init_data', 'quantize6', {'in': 4}),
                                 ('quantize6', 'quantize_data'),
                                 ('quantize_data', 'convolution', {'in': 1})],
                                nodes_with_edges_only=True)

        WeightsQuantize().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'convolution', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_negative_quantize_levels_None(self):
        graph = build_graph(nodes_attributes,
                            [('weights_const', 'weights_data'),
                             ('weights_data', 'quantize3', {'in': 0}),
                             ('input_low', 'input_low_data'),
                             ('input_low_data', 'quantize3', {'in': 1}),
                             ('input_high', 'input_high_data'),
                             ('input_high_data', 'quantize3', {'in': 2}),
                             ('output_low_init', 'output_low_init_data'),
                             ('output_low_init_data', 'quantize3', {'in': 3}),
                             ('output_high_init', 'output_high_init_data'),
                             ('output_high_init_data', 'quantize3', {'in': 4}),
                             ('quantize3', 'quantize_data'),
                             ('quantize_data', 'convolution', {'in': 1})],
                            nodes_with_edges_only=True)

        graph.graph['cmd_params'] = Namespace(keep_quantize_ops_in_IR=True)

        graph_ref = build_graph(nodes_attributes,
                                [('weights_const', 'weights_data'),
                                 ('weights_data', 'quantize3', {'in': 0}),
                                 ('input_low', 'input_low_data'),
                                 ('input_low_data', 'quantize3', {'in': 1}),
                                 ('input_high', 'input_high_data'),
                                 ('input_high_data', 'quantize3', {'in': 2}),
                                 ('output_low_init', 'output_low_init_data'),
                                 ('output_low_init_data', 'quantize3', {'in': 3}),
                                 ('output_high_init', 'output_high_init_data'),
                                 ('output_high_init_data', 'quantize3', {'in': 4}),
                                 ('quantize3', 'quantize_data'),
                                 ('quantize_data', 'convolution', {'in': 1})],
                                nodes_with_edges_only=True)

        WeightsQuantize().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'convolution', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_positive_quantize1(self):
        """
        int8 interval [0; 4]
        fp32 interval [-1.5; 3]
        """
        graph = build_graph(nodes_attributes,
                            [('weights_const', 'weights_data'),
                             ('weights_data', 'quantize1', {'in': 0}),
                             ('input_low', 'input_low_data'),
                             ('input_low_data', 'quantize1', {'in': 1}),
                             ('input_high', 'input_high_data'),
                             ('input_high_data', 'quantize1', {'in': 2}),
                             ('output_low_init', 'output_low_init_data'),
                             ('output_low_init_data', 'quantize1', {'in': 3}),
                             ('output_high_init', 'output_high_init_data'),
                             ('output_high_init_data', 'quantize1', {'in': 4}),
                             ('quantize1', 'quantize_data'),
                             ('quantize_data', 'convolution', {'in': 1}),
                             ('placeholder', 'placeholder_data'),
                             ('placeholder_data', 'convolution', {'in': 0})],
                            nodes_with_edges_only=True)
        graph.graph['cmd_params'] = Namespace(keep_quantize_ops_in_IR=True)

        graph_ref = build_graph(nodes_attributes,
                                [('weights_const', 'weights_data'),
                                 ('weights_data', 'new_quantize1', {'in': 0}),
                                 ('input_low', 'input_low_data'),
                                 ('input_low_data', 'new_quantize1', {'in': 1}),
                                 ('input_high', 'input_high_data'),
                                 ('input_high_data', 'new_quantize1', {'in': 2}),
                                 ('output_low', 'output_low_data'),
                                 ('output_low_data', 'new_quantize1', {'in': 3}),
                                 ('output_high', 'output_high_data'),
                                 ('output_high_data', 'new_quantize1', {'in': 4}),
                                 ('new_quantize1', 'new_quantize_data'),
                                 ('new_quantize_data', 'mul', {'in': 1}),
                                 ('scale', 'scale_data'),
                                 ('scale_data', 'mul', {'in': 0}),
                                 ('mul', 'mul_data'),
                                 ('mul_data', 'add', {'in': 1}),
                                 ('shift', 'shift_data'),
                                 ('shift_data', 'add', {'in': 0}),
                                 ('add', 'add_data'),
                                 ('add_data', 'quantize1', {'in': 0}),
                                 ('input_low_data', 'quantize1', {'in': 1}),
                                 ('input_high_data', 'quantize1', {'in': 2}),
                                 ('output_low_init', 'output_low_init_data'),
                                 ('output_low_init_data', 'quantize1', {'in': 3}),
                                 ('output_high_init', 'output_high_init_data'),
                                 ('output_high_init_data', 'quantize1', {'in': 4}),
                                 ('quantize1', 'quantize_data'),
                                 ('quantize_data', 'convolution', {'in': 1}),
                                 ('placeholder', 'placeholder_data'),
                                 ('placeholder_data', 'convolution', {'in': 0})],
                                nodes_with_edges_only=True)

        WeightsQuantize().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'convolution', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_positive_quantize2(self):
        """
        int8 interval [0; 121]
        fp32 interval [17.3; 256.1]
        """
        graph = build_graph(nodes_attributes,
                            [('weights_const', 'weights_data'),
                             ('weights_data', 'quantize4', {'in': 0}),
                             ('input_low', 'input_low_data'),
                             ('input_low_data', 'quantize4', {'in': 1}),
                             ('input_high', 'input_high_data'),
                             ('input_high_data', 'quantize4', {'in': 2}),
                             ('output_low_init', 'output_low_init_data1'),
                             ('output_low_init_data1', 'quantize4', {'in': 3}),
                             ('output_high_init', 'output_high_init_data1'),
                             ('output_high_init_data1', 'quantize4', {'in': 4}),
                             ('quantize4', 'quantize_data'),
                             ('quantize_data', 'convolution', {'in': 1}),
                             ('placeholder', 'placeholder_data'),
                             ('placeholder_data', 'convolution', {'in': 0})],
                            nodes_with_edges_only=True)
        graph.graph['cmd_params'] = Namespace(keep_quantize_ops_in_IR=True)

        graph_ref = build_graph(nodes_attributes,
                                [('weights_const', 'weights_data'),
                                 ('weights_data', 'new_quantize4', {'in': 0}),
                                 ('input_low', 'input_low_data'),
                                 ('input_low_data', 'new_quantize4', {'in': 1}),
                                 ('input_high', 'input_high_data'),
                                 ('input_high_data', 'new_quantize4', {'in': 2}),
                                 ('output_low', 'output_low_data'),
                                 ('output_low_data', 'new_quantize4', {'in': 3}),
                                 ('output_high', 'output_high_data'),
                                 ('output_high_data', 'new_quantize4', {'in': 4}),
                                 ('new_quantize4', 'new_quantize_data'),
                                 ('new_quantize_data', 'mul', {'in': 1}),
                                 ('scale1', 'scale_data'),
                                 ('scale_data', 'mul', {'in': 0}),
                                 ('mul', 'mul_data'),
                                 ('mul_data', 'add', {'in': 1}),
                                 ('shift1', 'shift_data'),
                                 ('shift_data', 'add', {'in': 0}),
                                 ('add', 'add_data'),
                                 ('add_data', 'quantize4', {'in': 0}),
                                 ('input_low_data', 'quantize4', {'in': 1}),
                                 ('input_high_data', 'quantize4', {'in': 2}),
                                 ('output_low_init', 'output_low_init_data1'),
                                 ('output_low_init_data1', 'quantize4', {'in': 3}),
                                 ('output_high_init', 'output_high_init_data1'),
                                 ('output_high_init_data1', 'quantize4', {'in': 4}),
                                 ('quantize4', 'quantize_data'),
                                 ('quantize_data', 'convolution', {'in': 1}),
                                 ('placeholder', 'placeholder_data'),
                                 ('placeholder_data', 'convolution', {'in': 0})],
                                nodes_with_edges_only=True)

        WeightsQuantize().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'convolution', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_positive_quantize3(self):
        """
        int8 interval [0; 201]
        fp32 interval [-2.573; -0.42]
        """

        graph = build_graph(nodes_attributes,
                            [('weights_const', 'weights_data'),
                             ('weights_data', 'quantize5', {'in': 0}),
                             ('input_low', 'input_low_data'),
                             ('input_low_data', 'quantize5', {'in': 1}),
                             ('input_high', 'input_high_data'),
                             ('input_high_data', 'quantize5', {'in': 2}),
                             ('output_low_init', 'output_low_init_data2'),
                             ('output_low_init_data2', 'quantize5', {'in': 3}),
                             ('output_high_init', 'output_high_init_data2'),
                             ('output_high_init_data2', 'quantize5', {'in': 4}),
                             ('quantize5', 'quantize_data'),
                             ('quantize_data', 'convolution', {'in': 1}),
                             ('placeholder', 'placeholder_data'),
                             ('placeholder_data', 'convolution', {'in': 0})],
                            nodes_with_edges_only=True)
        graph.graph['cmd_params'] = Namespace(keep_quantize_ops_in_IR=True)

        graph_ref = build_graph(nodes_attributes,
                                [('weights_const', 'weights_data'),
                                 ('weights_data', 'new_quantize5', {'in': 0}),
                                 ('input_low', 'input_low_data'),
                                 ('input_low_data', 'new_quantize5', {'in': 1}),
                                 ('input_high', 'input_high_data'),
                                 ('input_high_data', 'new_quantize5', {'in': 2}),
                                 ('output_low', 'output_low_data'),
                                 ('output_low_data', 'new_quantize5', {'in': 3}),
                                 ('output_high', 'output_high_data'),
                                 ('output_high_data', 'new_quantize5', {'in': 4}),
                                 ('new_quantize5', 'new_quantize_data'),
                                 ('new_quantize_data', 'mul', {'in': 1}),
                                 ('scale2', 'scale_data'),
                                 ('scale_data', 'mul', {'in': 0}),
                                 ('mul', 'mul_data'),
                                 ('mul_data', 'add', {'in': 1}),
                                 ('shift2', 'shift_data'),
                                 ('shift_data', 'add', {'in': 0}),
                                 ('add', 'add_data'),
                                 ('add_data', 'quantize5', {'in': 0}),
                                 ('input_low_data', 'quantize5', {'in': 1}),
                                 ('input_high_data', 'quantize5', {'in': 2}),
                                 ('output_low_init', 'output_low_init_data2'),
                                 ('output_low_init_data2', 'quantize5', {'in': 3}),
                                 ('output_high_init', 'output_high_init_data2'),
                                 ('output_high_init_data2', 'quantize5', {'in': 4}),
                                 ('quantize5', 'quantize_data'),
                                 ('quantize_data', 'convolution', {'in': 1}),
                                 ('placeholder', 'placeholder_data'),
                                 ('placeholder_data', 'convolution', {'in': 0})],
                                nodes_with_edges_only=True)

        WeightsQuantize().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'convolution', check_op_attrs=True)
        self.assertTrue(flag, resp)
