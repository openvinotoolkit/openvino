"""
 Copyright (c) 2020 Intel Corporation

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

import numpy as np
from generator import generator, generate

from extensions.back.compress_quantized_weights import CompressQuantizeWeights
from extensions.ops.fakequantize import FakeQuantize
from mo.front.common.partial_infer.eltwise import eltwise_infer
from mo.graph.graph import Node
from mo.ops.const import Const
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, regular_op_with_shaped_data, regular_op_with_empty_data, \
    valued_const_with_data, result, connect

nodes_attributes = {
    # placeholder
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {'kind': 'data'},

    # weights
    'weights_const': {'type': 'Const', 'kind': 'op', 'value': np.array([], dtype=np.float32), 'op': 'Const'},
    'weights_data': {'kind': 'data'},

    # quantize
    'quantize1': {'type': 'FakeQuantize', 'kind': 'op', 'levels': 5, 'op': 'FakeQuantize'},
    'quantize2': {'type': 'FakeQuantize', 'kind': 'op', 'levels': 2, 'op': 'FakeQuantize'},
    'quantize3': {'type': 'FakeQuantize', 'kind': 'op', 'levels': None, 'op': 'FakeQuantize'},
    'quantize4': {'type': 'FakeQuantize', 'kind': 'op', 'levels': 122, 'op': 'FakeQuantize'},
    'quantize5': {'type': 'FakeQuantize', 'kind': 'op', 'levels': 202, 'op': 'FakeQuantize'},
    'quantize6': {'type': 'FakeQuantize', 'kind': 'op', 'levels': 257, 'op': 'FakeQuantize'},
    'quantize_data': {'kind': 'data'},
    'new_quantize1': {'kind': 'op', 'type': 'FakeQuantize', 'levels': 5, 'op': 'FakeQuantize'},
    'new_quantize4': {'kind': 'op', 'type': 'FakeQuantize', 'levels': 122, 'op': 'FakeQuantize'},
    'new_quantize5': {'kind': 'op', 'type': 'FakeQuantize', 'levels': 202, 'op': 'FakeQuantize'},
    'new_quantize_data': {'kind': 'data'},

    #  quantize input/output
    'output_high_init': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'output_high_init_data': {'kind': 'data', 'value': 3},
    'output_low_init': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'output_low_init_data': {'kind': 'data', 'value': -1.5},

    'input_low': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'input_low_data': {'kind': 'data'},
    'input_high': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'input_high_data': {'kind': 'data'},

    'output_low': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'output_low_data': {'kind': 'data'},
    'output_high': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'output_high_data': {'kind': 'data'},

    'output_high_init_data1': {'kind': 'data', 'value': 256.1},
    'output_low_init_data1': {'kind': 'data', 'value': 17.3},

    'output_high_init_data2': {'kind': 'data', 'value': -0.42},
    'output_low_init_data2': {'kind': 'data', 'value': -2.573},

    # eltwise ops
    'mul': {'kind': 'op', 'op': 'Mul'},
    'add': {'kind': 'op', 'op': 'Add'},

    'scale': {'kind': 'op', 'type': 'Const', 'value': 1.125, 'op': 'Const'},
    'shift': {'kind': 'op', 'type': 'Const', 'value': -1.5, 'op': 'Const'},
    'scale1': {'kind': 'op', 'type': 'Const', 'value': 1.9735537190082646, 'op': 'Const'},
    'shift1': {'kind': 'op', 'type': 'Const', 'value': 17.3, 'op': 'Const'},
    'scale2': {'kind': 'op', 'type': 'Const', 'value': 0.010711442786069652, 'op': 'Const'},
    'shift2': {'kind': 'op', 'type': 'Const', 'value': -2.573, 'op': 'Const'},

    'shift_data': {'kind': 'data'},
    'scale_data': {'kind': 'data'},
    'mul_data': {'kind': 'data'},
    'add_data': {'kind': 'data'},

    'convolution': {'type': 'Convolution', 'kind': 'op', 'op': 'Convolution'},
    'convert': {'type': 'Convert', 'kind': 'op', 'dst_type': np.float32, 'op': 'Cast'},
    'convert_data': {'kind': 'data'},

    'result_data': {'kind': 'data'},
    'result': {'kind': 'op', 'op': 'Result'},

    # accuracy test
    'ac_weights': {'kind': 'op', 'op': 'Const', 'shape': None, 'value': None, 'infer': Const.infer},
    'ac_weights_data': {'kind': 'data', 'shape': None, 'value': None},

    'ac_input_low': {'kind': 'op', 'type': 'Const', 'shape': None, 'value': None, 'infer': Const.infer, 'op': 'Const'},
    'ac_input_low_data': {'kind': 'data', 'value': None, 'shape': None},
    'ac_input_high': {'kind': 'op', 'type': 'Const', 'shape': None, 'value': None, 'infer': Const.infer, 'op': 'Const'},
    'ac_input_high_data': {'kind': 'data', 'value': None, 'shape': None},
    'ac_output_low': {'kind': 'op', 'type': 'Const', 'shape': None, 'value': None, 'infer': Const.infer, 'op': 'Const'},
    'ac_output_low_data': {'kind': 'data', 'value': None, 'shape': None},
    'ac_output_high': {'kind': 'op', 'type': 'Const', 'shape': None, 'value': None, 'infer': Const.infer, 'op': 'Const'},
    'ac_output_high_data': {'kind': 'data', 'value': None, 'shape': None},

    'ac_fakeQuantize': {'kind': 'op', 'type': 'FakeQuantize', 'levels': None, 'infer': FakeQuantize.infer, 'op': 'FakeQuantize'},
    'ac_fakeQuantize_data': {'kind': 'data', 'shape': None, 'value': None},
    'ac_quantize': {'kind': 'op', 'type': 'fakeQuantize', 'levels': None, 'infer': FakeQuantize.infer, 'op': 'FakeQuantize'},
    'ac_quantize_data': {'kind': 'data', 'shape': None, 'value': None},

    'ac_convolution': {'kind': 'op', 'type': 'Convolution', 'op': 'Convolution'},

    'ac_mul': {'kind': 'op', 'op': 'Mul', 'infer': lambda node: eltwise_infer(node, lambda a, b: a * b)},
    'ac_mul_data': {'kind': 'data', 'shape': None, 'value': None},
    'ac_add': {'kind': 'op', 'op': 'Add', 'infer': lambda node: eltwise_infer(node, lambda a, b: a + b)},
    'ac_add_data': {'kind': 'data', 'shape': None, 'value': None},

    'ac_scale': {'kind': 'op', 'type': 'Const', 'shape': None, 'value': None, 'infer': Const.infer, 'op': 'Const'},
    'ac_scale_data': {'kind': 'data', 'shape': None, 'value': None},
    'ac_shift': {'kind': 'op', 'type': 'Const', 'shape': None, 'value': None, 'infer': Const.infer, 'op': 'Const'},
    'ac_shift_data': {'kind': 'data', 'shape': None, 'value': None},

    'ac_output_low_ref': {'kind': 'op', 'type': 'Const', 'shape': None, 'value': None, 'infer': Const.infer, 'op': 'Const'},
    'ac_output_low_ref_data': {'kind': 'data', 'shape': None, 'value': None},
    'ac_output_high_ref': {'kind': 'op', 'type': 'Const', 'shape': None, 'value': None, 'infer': Const.infer, 'op': 'Const'},
    'ac_output_high_ref_data': {'kind': 'data', 'shape': None, 'value': None}
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
        graph.graph['cmd_params'] = Namespace(data_type='FP32')

        graph_ref = build_graph(nodes_attributes,
                                [('weights_const', 'weights_data'),
                                 ('weights_data', 'mul'),
                                 ('scale', 'mul'),
                                 ('mul', 'add'),
                                 ('shift', 'add'),
                                 ('add', 'quantize_data'),
                                 ('quantize_data', 'convolution')],
                                nodes_with_edges_only=True)

        CompressQuantizeWeights().find_and_replace_pattern(graph)
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

        graph.graph['cmd_params'] = Namespace(data_type='FP32')

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

        CompressQuantizeWeights().find_and_replace_pattern(graph)
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

        graph.graph['cmd_params'] = Namespace(data_type='FP32')

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

        CompressQuantizeWeights().find_and_replace_pattern(graph)
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

        graph.graph['cmd_params'] = Namespace(data_type='FP32')

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

        CompressQuantizeWeights().find_and_replace_pattern(graph)
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
                            {'input_low': {'shape': np.array([1]), 'value': -1.5},
                             'input_low_data': {'value': -1.5},
                             'input_high': {'shape': np.array([1]), 'value': 3},
                             'input_high_data': {'value': 3}},
                            nodes_with_edges_only=True)
        graph.graph['cmd_params'] = Namespace(data_type='FP32')

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
                                 ('new_quantize_data', 'convert'),
                                 ('convert', 'convert_data'),
                                 ('convert_data', 'quantize1', {'in': 0}),
                                 ('output_low_data', 'quantize1', {'in': 1}),
                                 ('output_high_data', 'quantize1', {'in': 2}),
                                 ('output_low_init', 'output_low_init_data'),
                                 ('output_low_init_data', 'quantize1', {'in': 3}),
                                 ('output_high_init', 'output_high_init_data'),
                                 ('output_high_init_data', 'quantize1', {'in': 4}),
                                 ('quantize1', 'quantize_data'),
                                 ('quantize_data', 'convolution', {'in': 1}),
                                 ('placeholder', 'placeholder_data'),
                                 ('placeholder_data', 'convolution', {'in': 0})],
                                nodes_with_edges_only=True)

        CompressQuantizeWeights().find_and_replace_pattern(graph)
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
                            {'input_low': {'shape': np.array([1]), 'value': 17.3},
                             'input_low_data': {'value': 17.3},
                             'input_high': {'shape': np.array([1]), 'value': 256.1},
                             'input_high_data': {'value': 256.1}},
                            nodes_with_edges_only=True)
        graph.graph['cmd_params'] = Namespace(data_type='FP32')

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
                                 ('new_quantize_data', 'convert'),
                                 ('convert', 'convert_data'),
                                 ('convert_data', 'quantize4', {'in': 0}),
                                 ('output_low_data', 'quantize4', {'in': 1}),
                                 ('output_high_data', 'quantize4', {'in': 2}),
                                 ('output_low_init', 'output_low_init_data1'),
                                 ('output_low_init_data1', 'quantize4', {'in': 3}),
                                 ('output_high_init', 'output_high_init_data1'),
                                 ('output_high_init_data1', 'quantize4', {'in': 4}),
                                 ('quantize4', 'quantize_data'),
                                 ('quantize_data', 'convolution', {'in': 1}),
                                 ('placeholder', 'placeholder_data'),
                                 ('placeholder_data', 'convolution', {'in': 0})],
                                nodes_with_edges_only=True)

        CompressQuantizeWeights().find_and_replace_pattern(graph)
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
                            {'input_low': {'shape': np.array([1]), 'value': -2.573},
                             'input_low_data': {'value': -2.573},
                             'input_high': {'shape': np.array([1]), 'value': -0.42},
                             'input_high_data': {'value': -0.42}},
                            nodes_with_edges_only=True)
        graph.graph['cmd_params'] = Namespace(data_type='FP32')

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
                                 ('new_quantize_data', 'convert'),
                                 ('convert', 'convert_data'),
                                 ('convert_data', 'quantize5', {'in': 0}),
                                 ('output_low_data', 'quantize5', {'in': 1}),
                                 ('output_high_data', 'quantize5', {'in': 2}),
                                 ('output_low_init', 'output_low_init_data2'),
                                 ('output_low_init_data2', 'quantize5', {'in': 3}),
                                 ('output_high_init', 'output_high_init_data2'),
                                 ('output_high_init_data2', 'quantize5', {'in': 4}),
                                 ('quantize5', 'quantize_data'),
                                 ('quantize_data', 'convolution', {'in': 1}),
                                 ('placeholder', 'placeholder_data'),
                                 ('placeholder_data', 'convolution', {'in': 0})],
                                nodes_with_edges_only=True)

        CompressQuantizeWeights().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'convolution', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_accuracy_tensor1(self):
        """
        [1.0, 2.0, 3.0, 4.0]
        """

        graph = build_graph(nodes_attributes,
                            [('ac_weights', 'ac_weights_data'),
                             ('ac_weights_data', 'ac_fakeQuantize', {'in': 0}),
                             ('ac_input_low', 'ac_input_low_data'),
                             ('ac_input_low_data', 'ac_fakeQuantize', {'in': 1}),
                             ('ac_input_high', 'ac_input_high_data'),
                             ('ac_input_high_data', 'ac_fakeQuantize', {'in': 2}),
                             ('ac_output_low', 'ac_output_low_data'),
                             ('ac_output_low_data', 'ac_fakeQuantize', {'in': 3}),
                             ('ac_output_high', 'ac_output_high_data'),
                             ('ac_output_high_data', 'ac_fakeQuantize', {'in': 4}),
                             ('ac_fakeQuantize', 'ac_fakeQuantize_data'),
                             ('ac_fakeQuantize_data', 'ac_convolution', {'in': 1}),
                             ('placeholder', 'placeholder_data'),
                             ('placeholder_data', 'ac_convolution', {'in': 0}),
                             ('ac_convolution', 'result_data'),
                             ('result_data', 'result')
                             ],
                            {'ac_weights': {'shape': np.array([4]), 'value': np.array([1.0, 2.0, 3.0, 4.0])},
                             'ac_input_low': {'shape': np.array([1]), 'value': 1},
                             'ac_input_high': {'shape': np.array([1]), 'value': 4},
                             'ac_output_low': {'shape': np.array([1]), 'value': 1},
                             'ac_output_high': {'shape': np.array([1]), 'value': 4},
                             'ac_fakeQuantize': {'levels': 256}},
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                            [('ac_weights', 'ac_weights_data'),
                             ('ac_weights_data', 'ac_quantize', {'in': 0}),
                             ('ac_input_low', 'ac_input_low_data'),
                             ('ac_input_low_data', 'ac_quantize', {'in': 1}),
                             ('ac_input_high', 'ac_input_high_data'),
                             ('ac_input_high_data', 'ac_quantize', {'in': 2}),
                             ('ac_output_low_ref', 'ac_output_low_ref_data'),
                             ('ac_output_low_ref_data', 'ac_quantize', {'in': 3}),
                             ('ac_output_high_ref', 'ac_output_high_ref_data'),
                             ('ac_output_high_ref_data', 'ac_quantize', {'in': 4}),
                             ('ac_quantize', 'ac_quantize_data'),
                             ('ac_quantize_data', 'ac_mul', {'in': 1}),
                             ('ac_scale', 'ac_scale_data'),
                             ('ac_scale_data', 'ac_mul', {'in': 0}),
                             ('ac_mul', 'ac_mul_data'),
                             ('ac_mul_data', 'ac_add', {'in': 1}),
                             ('ac_shift', 'ac_shift_data'),
                             ('ac_shift_data', 'ac_add', {'in': 0}),
                             ('ac_add', 'ac_add_data'),
                             ('ac_add_data', 'ac_fakeQuantize', {'in': 0}),
                             ('ac_input_low', 'ac_input_low_data'),
                             ('ac_input_low_data', 'ac_fakeQuantize', {'in': 1}),
                             ('ac_input_high', 'ac_input_high_data'),
                             ('ac_input_high_data', 'ac_fakeQuantize', {'in': 2}),
                             ('ac_output_low', 'ac_output_low_data'),
                             ('ac_output_low_data', 'ac_fakeQuantize', {'in': 3}),
                             ('ac_output_high', 'ac_output_high_data'),
                             ('ac_output_high_data', 'ac_fakeQuantize', {'in': 4}),
                             ('ac_fakeQuantize', 'ac_fakeQuantize_data'),
                             ('ac_fakeQuantize_data', 'ac_convolution', {'in': 1}),
                             ('placeholder', 'placeholder_data'),
                             ('placeholder_data', 'ac_convolution', {'in': 0}),
                             ('ac_convolution', 'result_data'),
                             ('result_data', 'result')
                             ],
                            {'ac_weights': {'shape': np.array([4]), 'value': np.array([1.0, 2.0, 3.0, 4.0])},
                             'ac_quantize': {'levels': 256},
                             'ac_fakeQuantize': {'levels': 256},
                             'ac_input_low': {'shape': np.array([1]), 'value': 1},
                             'ac_input_high': {'shape': np.array([1]), 'value': 4},
                             'ac_output_low_ref': {'shape': np.array([1]), 'value': 0},
                             'ac_output_high_ref': {'shape': np.array([1]), 'value': 255},
                             'ac_scale': {'shape': np.array([1]), 'value': 0.011764705882352941},
                             'ac_shift': {'shape': np.array([1]), 'value': 1},
                             'ac_output_low': {'shape': np.array([1]), 'value': 1},
                             'ac_output_high': {'shape': np.array([1]), 'value': 4},
                             },
                            nodes_with_edges_only=True)

        graph.graph['cmd_params'] = Namespace(data_type='FP32', keep_shape_ops=True)
        graph_ref.graph['cmd_params'] = Namespace(keep_shape_ops=True)

        graph.clean_up()
        graph_ref.clean_up()

        w_array = Node(graph, 'ac_weights').out_port(0).get_destination().data.get_value()
        w_array_ref = Node(graph_ref, 'ac_weights').out_port(0).get_destination().data.get_value()

        self.assertTrue(np.all(w_array == w_array_ref))

    def test_accuracy_tensor2(self):

        """
        [-1.5, -0.32, 0.167, 2.8]
        """

        graph = build_graph(nodes_attributes,
                            [('ac_weights', 'ac_weights_data'),
                             ('ac_weights_data', 'ac_fakeQuantize', {'in': 0}),
                             ('ac_input_low', 'ac_input_low_data'),
                             ('ac_input_low_data', 'ac_fakeQuantize', {'in': 1}),
                             ('ac_input_high', 'ac_input_high_data'),
                             ('ac_input_high_data', 'ac_fakeQuantize', {'in': 2}),
                             ('ac_output_low', 'ac_output_low_data'),
                             ('ac_output_low_data', 'ac_fakeQuantize', {'in': 3}),
                             ('ac_output_high', 'ac_output_high_data'),
                             ('ac_output_high_data', 'ac_fakeQuantize', {'in': 4}),
                             ('ac_fakeQuantize', 'ac_fakeQuantize_data'),
                             ('ac_fakeQuantize_data', 'ac_convolution', {'in': 1}),
                             ('placeholder', 'placeholder_data'),
                             ('placeholder_data', 'ac_convolution', {'in': 0}),
                             ('ac_convolution', 'result_data'),
                             ('result_data', 'result')
                             ],
                            {'ac_weights': {'shape': np.array([4]), 'value': np.array([-1.5, -0.32, 0.167, 2.8])},
                             'ac_input_low': {'shape': np.array([1]), 'value': -1.5},
                             'ac_input_high': {'shape': np.array([1]), 'value': 2.8},
                             'ac_output_low': {'shape': np.array([1]), 'value': -1.5},
                             'ac_output_high': {'shape': np.array([1]), 'value': 2.8},
                             'ac_fakeQuantize': {'levels': 256}},
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('ac_weights', 'ac_weights_data'),
                                 ('ac_weights_data', 'ac_quantize', {'in': 0}),
                                 ('ac_input_low', 'ac_input_low_data'),
                                 ('ac_input_low_data', 'ac_quantize', {'in': 1}),
                                 ('ac_input_high', 'ac_input_high_data'),
                                 ('ac_input_high_data', 'ac_quantize', {'in': 2}),
                                 ('ac_output_low_ref', 'ac_output_low_ref_data'),
                                 ('ac_output_low_ref_data', 'ac_quantize', {'in': 3}),
                                 ('ac_output_high_ref', 'ac_output_high_ref_data'),
                                 ('ac_output_high_ref_data', 'ac_quantize', {'in': 4}),
                                 ('ac_quantize', 'ac_quantize_data'),
                                 ('ac_quantize_data', 'ac_mul', {'in': 1}),
                                 ('ac_scale', 'ac_scale_data'),
                                 ('ac_scale_data', 'ac_mul', {'in': 0}),
                                 ('ac_mul', 'ac_mul_data'),
                                 ('ac_mul_data', 'ac_add', {'in': 1}),
                                 ('ac_shift', 'ac_shift_data'),
                                 ('ac_shift_data', 'ac_add', {'in': 0}),
                                 ('ac_add', 'ac_add_data'),
                                 ('ac_add_data', 'ac_fakeQuantize', {'in': 0}),
                                 ('ac_input_low', 'ac_input_low_data'),
                                 ('ac_input_low_data', 'ac_fakeQuantize', {'in': 1}),
                                 ('ac_input_high', 'ac_input_high_data'),
                                 ('ac_input_high_data', 'ac_fakeQuantize', {'in': 2}),
                                 ('ac_output_low', 'ac_output_low_data'),
                                 ('ac_output_low_data', 'ac_fakeQuantize', {'in': 3}),
                                 ('ac_output_high', 'ac_output_high_data'),
                                 ('ac_output_high_data', 'ac_fakeQuantize', {'in': 4}),
                                 ('ac_fakeQuantize', 'ac_fakeQuantize_data'),
                                 ('ac_fakeQuantize_data', 'ac_convolution', {'in': 1}),
                                 ('placeholder', 'placeholder_data'),
                                 ('placeholder_data', 'ac_convolution', {'in': 0}),
                                 ('ac_convolution', 'result_data'),
                                 ('result_data', 'result')
                                 ],
                                {'ac_weights': {'shape': np.array([4]), 'value': np.array([-1.5, -0.32, 0.167, 2.8])},
                                 'ac_quantize': {'levels': 256},
                                 'ac_fakeQuantize': {'levels': 256},
                                 'ac_input_low': {'shape': np.array([1]), 'value': -1.5},
                                 'ac_input_high': {'shape': np.array([1]), 'value': 2.8},
                                 'ac_output_low_ref': {'shape': np.array([1]), 'value': 0},
                                 'ac_output_high_ref': {'shape': np.array([1]), 'value': 255},
                                 'ac_scale': {'shape': np.array([1]), 'value': 0.016862745098039214},
                                 'ac_shift': {'shape': np.array([1]), 'value': -1.5},
                                 'ac_output_low': {'shape': np.array([1]), 'value': -1.5},
                                 'ac_output_high': {'shape': np.array([1]), 'value': 2.8},
                                 },
                                nodes_with_edges_only=True)

        graph.graph['cmd_params'] = Namespace(data_type='FP32', keep_shape_ops=True)
        graph_ref.graph['cmd_params'] = Namespace(keep_shape_ops=True)

        graph.clean_up()
        graph_ref.clean_up()

        w_array = Node(graph, 'ac_weights').out_port(0).get_destination().data.get_value()
        w_array_ref = Node(graph_ref, 'ac_weights').out_port(0).get_destination().data.get_value()

        self.assertTrue(np.all(w_array == w_array_ref))

    def test_accuracy_tensor3(self):

        """
        [-2.586, -1.338, 2.773, 4.414]
        """

        graph = build_graph(nodes_attributes,
                            [('ac_weights', 'ac_weights_data'),
                             ('ac_weights_data', 'ac_fakeQuantize', {'in': 0}),
                             ('ac_input_low', 'ac_input_low_data'),
                             ('ac_input_low_data', 'ac_fakeQuantize', {'in': 1}),
                             ('ac_input_high', 'ac_input_high_data'),
                             ('ac_input_high_data', 'ac_fakeQuantize', {'in': 2}),
                             ('ac_output_low', 'ac_output_low_data'),
                             ('ac_output_low_data', 'ac_fakeQuantize', {'in': 3}),
                             ('ac_output_high', 'ac_output_high_data'),
                             ('ac_output_high_data', 'ac_fakeQuantize', {'in': 4}),
                             ('ac_fakeQuantize', 'ac_fakeQuantize_data'),
                             ('ac_fakeQuantize_data', 'ac_convolution', {'in': 1}),
                             ('placeholder', 'placeholder_data'),
                             ('placeholder_data', 'ac_convolution', {'in': 0}),
                             ('ac_convolution', 'result_data'),
                             ('result_data', 'result')],
                            {'ac_weights': {'shape': np.array([4]), 'value': np.array([-2.586, -1.338, 2.773, 4.414])},
                             'ac_input_low': {'shape': np.array([1]), 'value': -2.586},
                             'ac_input_high': {'shape': np.array([1]), 'value': 4.414},
                             'ac_output_low': {'shape': np.array([1]), 'value': -2.586},
                             'ac_output_high': {'shape': np.array([1]), 'value': 4.414},
                             'ac_fakeQuantize': {'levels': 256}},
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('ac_weights', 'ac_weights_data'),
                                 ('ac_weights_data', 'ac_quantize', {'in': 0}),
                                 ('ac_input_low', 'ac_input_low_data'),
                                 ('ac_input_low_data', 'ac_quantize', {'in': 1}),
                                 ('ac_input_high', 'ac_input_high_data'),
                                 ('ac_input_high_data', 'ac_quantize', {'in': 2}),
                                 ('ac_output_low_ref', 'ac_output_low_ref_data'),
                                 ('ac_output_low_ref_data', 'ac_quantize', {'in': 3}),
                                 ('ac_output_high_ref', 'ac_output_high_ref_data'),
                                 ('ac_output_high_ref_data', 'ac_quantize', {'in': 4}),
                                 ('ac_quantize', 'ac_quantize_data'),
                                 ('ac_quantize_data', 'ac_mul', {'in': 1}),
                                 ('ac_scale', 'ac_scale_data'),
                                 ('ac_scale_data', 'ac_mul', {'in': 0}),
                                 ('ac_mul', 'ac_mul_data'),
                                 ('ac_mul_data', 'ac_add', {'in': 1}),
                                 ('ac_shift', 'ac_shift_data'),
                                 ('ac_shift_data', 'ac_add', {'in': 0}),
                                 ('ac_add', 'ac_add_data'),
                                 ('ac_add_data', 'ac_fakeQuantize', {'in': 0}),
                                 ('ac_input_low', 'ac_input_low_data'),
                                 ('ac_input_low_data', 'ac_fakeQuantize', {'in': 1}),
                                 ('ac_input_high', 'ac_input_high_data'),
                                 ('ac_input_high_data', 'ac_fakeQuantize', {'in': 2}),
                                 ('ac_output_low', 'ac_output_low_data'),
                                 ('ac_output_low_data', 'ac_fakeQuantize', {'in': 3}),
                                 ('ac_output_high', 'ac_output_high_data'),
                                 ('ac_output_high_data', 'ac_fakeQuantize', {'in': 4}),
                                 ('ac_fakeQuantize', 'ac_fakeQuantize_data'),
                                 ('ac_fakeQuantize_data', 'ac_convolution', {'in': 1}),
                                 ('placeholder', 'placeholder_data'),
                                 ('placeholder_data', 'ac_convolution', {'in': 0}),
                                 ('ac_convolution', 'result_data'),
                                 ('result_data', 'result')
                                 ],
                                {'ac_weights': {'shape': np.array([4]), 'value': np.array([-2.586, -1.338, 2.773,
                                                                                           4.414])},
                                 'ac_quantize': {'levels': 256},
                                 'ac_fakeQuantize': {'levels': 256},
                                 'ac_input_low': {'shape': np.array([1]), 'value': -2.586},
                                 'ac_input_high': {'shape': np.array([1]), 'value': 4.414},
                                 'ac_output_low_ref': {'shape': np.array([1]), 'value': 0},
                                 'ac_output_high_ref': {'shape': np.array([1]), 'value': 255},
                                 'ac_scale': {'shape': np.array([1]), 'value': 0.027450980392156862},
                                 'ac_shift': {'shape': np.array([1]), 'value': -2.586},
                                 'ac_output_low': {'shape': np.array([1]), 'value': -2.586},
                                 'ac_output_high': {'shape': np.array([1]), 'value': 4.414},
                                 },
                                nodes_with_edges_only=True)

        graph.graph['cmd_params'] = Namespace(data_type='FP32', keep_shape_ops=True)
        graph_ref.graph['cmd_params'] = Namespace(keep_shape_ops=True)

        graph.clean_up()
        graph_ref.clean_up()

        w_array = Node(graph, 'ac_weights').out_port(0).get_destination().data.get_value()
        w_array_ref = Node(graph_ref, 'ac_weights').out_port(0).get_destination().data.get_value()

        self.assertTrue(np.all(w_array == w_array_ref))

    def test_accuracy_tensor4(self):

        eps = np.finfo(np.float32).eps

        graph = build_graph(nodes_attributes,
                            [('ac_weights', 'ac_weights_data'),
                             ('ac_weights_data', 'ac_fakeQuantize', {'in': 0}),
                             ('ac_input_low', 'ac_input_low_data'),
                             ('ac_input_low_data', 'ac_fakeQuantize', {'in': 1}),
                             ('ac_input_high', 'ac_input_high_data'),
                             ('ac_input_high_data', 'ac_fakeQuantize', {'in': 2}),
                             ('ac_output_low', 'ac_output_low_data'),
                             ('ac_output_low_data', 'ac_fakeQuantize', {'in': 3}),
                             ('ac_output_high', 'ac_output_high_data'),
                             ('ac_output_high_data', 'ac_fakeQuantize', {'in': 4}),
                             ('ac_fakeQuantize', 'ac_fakeQuantize_data'),
                             ('ac_fakeQuantize_data', 'ac_convolution', {'in': 1}),
                             ('placeholder', 'placeholder_data'),
                             ('placeholder_data', 'ac_convolution', {'in': 0}),
                             ('ac_convolution', 'result_data'),
                             ('result_data', 'result')],
                            {'ac_weights': {'shape': np.array([4]), 'value': np.array([1, 1 + eps,
                                                                                       1 + 2 * eps, 1 + 3 * eps])},
                             'ac_input_low': {'shape': np.array([1]), 'value': 1},
                             'ac_input_high': {'shape': np.array([1]), 'value': 1 + 3 * eps},
                             'ac_output_low': {'shape': np.array([1]), 'value': 1},
                             'ac_output_high': {'shape': np.array([1]), 'value': 1 + 3 * eps},
                             'ac_fakeQuantize': {'levels': 256}},
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('ac_weights', 'ac_weights_data'),
                                 ('ac_weights_data', 'ac_quantize', {'in': 0}),
                                 ('ac_input_low', 'ac_input_low_data'),
                                 ('ac_input_low_data', 'ac_quantize', {'in': 1}),
                                 ('ac_input_high', 'ac_input_high_data'),
                                 ('ac_input_high_data', 'ac_quantize', {'in': 2}),
                                 ('ac_output_low_ref', 'ac_output_low_ref_data'),
                                 ('ac_output_low_ref_data', 'ac_quantize', {'in': 3}),
                                 ('ac_output_high_ref', 'ac_output_high_ref_data'),
                                 ('ac_output_high_ref_data', 'ac_quantize', {'in': 4}),
                                 ('ac_quantize', 'ac_quantize_data'),
                                 ('ac_quantize_data', 'ac_mul', {'in': 1}),
                                 ('ac_scale', 'ac_scale_data'),
                                 ('ac_scale_data', 'ac_mul', {'in': 0}),
                                 ('ac_mul', 'ac_mul_data'),
                                 ('ac_mul_data', 'ac_add', {'in': 1}),
                                 ('ac_shift', 'ac_shift_data'),
                                 ('ac_shift_data', 'ac_add', {'in': 0}),
                                 ('ac_add', 'ac_add_data'),
                                 ('ac_add_data', 'ac_fakeQuantize', {'in': 0}),
                                 ('ac_input_low', 'ac_input_low_data'),
                                 ('ac_input_low_data', 'ac_fakeQuantize', {'in': 1}),
                                 ('ac_input_high', 'ac_input_high_data'),
                                 ('ac_input_high_data', 'ac_fakeQuantize', {'in': 2}),
                                 ('ac_output_low', 'ac_output_low_data'),
                                 ('ac_output_low_data', 'ac_fakeQuantize', {'in': 3}),
                                 ('ac_output_high', 'ac_output_high_data'),
                                 ('ac_output_high_data', 'ac_fakeQuantize', {'in': 4}),
                                 ('ac_fakeQuantize', 'ac_fakeQuantize_data'),
                                 ('ac_fakeQuantize_data', 'ac_convolution', {'in': 1}),
                                 ('placeholder', 'placeholder_data'),
                                 ('placeholder_data', 'ac_convolution', {'in': 0}),
                                 ('ac_convolution', 'result_data'),
                                 ('result_data', 'result')],
                                {'ac_weights': {'shape': np.array([4]), 'value': np.array([1, 1 + eps,
                                                                                       1 + 2 * eps, 1 + 3 * eps])},
                                 'ac_quantize': {'levels': 256},
                                 'ac_fakeQuantize': {'levels': 256},
                                 'ac_input_low': {'shape': np.array([1]), 'value': 1},
                                 'ac_input_high': {'shape': np.array([1]), 'value': 1 + 3 * eps},
                                 'ac_output_low_ref': {'shape': np.array([1]), 'value': 0},
                                 'ac_output_high_ref': {'shape': np.array([1]), 'value': 255},
                                 'ac_scale': {'shape': np.array([1]), 'value': 3 * eps / 255},
                                 'ac_shift': {'shape': np.array([1]), 'value': 1},
                                 'ac_output_low': {'shape': np.array([1]), 'value': 1},
                                 'ac_output_high': {'shape': np.array([1]), 'value': 1 + 3 * eps},
                                 },
                                nodes_with_edges_only=True)

        graph.graph['cmd_params'] = Namespace(data_type='FP32', keep_shape_ops=True)
        graph_ref.graph['cmd_params'] = Namespace(keep_shape_ops=True)

        graph.clean_up()
        graph_ref.clean_up()

        w_array = Node(graph, 'ac_weights').out_port(0).get_destination().data.get_value()
        w_array_ref = Node(graph_ref, 'ac_weights').out_port(0).get_destination().data.get_value()

        self.assertTrue(np.all(w_array == w_array_ref))


@generator
class CompressionDataTypeTest(unittest.TestCase):
    @generate(*[
        ('FP32', np.int64),
        ('FP16', np.int64),
        ('FP32', np.int32),
        ('FP16', np.int32),
        ('FP32', np.float64, np.float32),
        ('FP16', np.float64, np.float16),
        ('FP32', np.float32, np.float32),
        ('FP16', np.float32, np.float16),
        ('FP32', np.float16, np.float32),
        ('FP16', np.float16, np.float16),
        ])
    def test_data_type(self, model_dtype, original, transformed=None):
        if transformed is None:
            transformed = original

        nodes = {
            **valued_const_with_data('weights', np.ones([1, 2, 3, 4], dtype=original)),

            **valued_const_with_data('int_weights', np.ones([1, 2, 3, 4], dtype=np.uint8)),
            **regular_op_with_shaped_data('cast', [1, 2, 3, 4], {'type': 'Convert', 'dst_type': transformed,
                                                                 'op': 'Cast'}),

            **valued_const_with_data('il', np.array([0])),
            **valued_const_with_data('ih', np.array([254])),
            **valued_const_with_data('ol', np.array([0])),
            **valued_const_with_data('oh', np.array([254])),

            **regular_op_with_shaped_data('FQ', [1, 2, 3, 4], {'type': 'FakeQuantize', 'infer': FakeQuantize.infer,
                                                               'stop_value_propagation': True, 'levels': 255,
                                                               'op': 'FakeQuantize'}),
            **result()
        }

        graph = build_graph(nodes, [
            *connect('weights:0', '0:FQ'),
            *connect('il:0', '1:FQ'),
            *connect('ih:0', '2:FQ'),
            *connect('ol:0', '3:FQ'),
            *connect('oh:0', '4:FQ'),
            *connect('FQ:0', 'output'),
        ], nodes_with_edges_only=True)
        graph.graph['cmd_params'] = Namespace(data_type=model_dtype, keep_shape_ops=True)

        CompressQuantizeWeights().find_and_replace_pattern(graph)
        graph.clean_up()

        graph_ref = build_graph(nodes, [
            *connect('int_weights:0', '0:cast'),
            *connect('cast:0', '0:FQ'),
            *connect('il:0', '1:FQ'),
            *connect('ih:0', '2:FQ'),
            *connect('ol:0', '3:FQ'),
            *connect('oh:0', '4:FQ'),
            *connect('FQ:0', 'output'),
        ], nodes_with_edges_only=True)
        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
