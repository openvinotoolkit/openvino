"""
 Copyright (C) 2018-2020 Intel Corporation

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

import numpy as np

from mo.front.common.partial_infer.eltwise import eltwise_infer
from mo.graph.graph import Node
from mo.middle.passes.fusing.fuse_linear_ops import _fuse_mul, _fuse_add, fuse_linear_ops
from mo.utils.unittest.graph import build_graph
from mo.utils.ir_engine.compare_graphs import compare_graphs

nodes_attributes = {
    'placeholder_1': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # ScaleShift layer
    'scaleshift_1': {'type': 'ScaleShift', 'kind': 'op', 'op': 'ScaleShift'},
    'scaleshift_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'scaleshift_1_b': {'value': None, 'shape': None, 'kind': 'data'},
    'const_scaleshift_1_w': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None},
    'const_scaleshift_1_b': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None},
    'scaleshift_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Mul and Add operations
    'mul_1': {'type': 'Mul', 'kind': 'op', 'op': 'Mul', 'can_be_fused': True,
              'infer': lambda node: eltwise_infer(node, lambda a, b: a*b)},
    'mul_1_w': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'const_mul_1_w': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None},
    'mul_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'add_1': {'type': 'Add', 'kind': 'op', 'op': 'Add', 'can_be_fused': True},
    'add_1_w': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'const_add_1_w': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None},
    'add_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # Mul2 and Add2 operations
    'mul_2': {'type': 'Mul', 'kind': 'op', 'op': 'Mul', 'can_be_fused': True},
    'mul_2_w': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'const_mul_2_w': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None},
    'mul_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'add_2': {'type': 'Add', 'kind': 'op', 'op': 'Add', 'can_be_fused': True},
    'add_2_w': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'const_add_2_w': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None},
    'add_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # Concat1 operation
    'concat_1': {'type': 'Concat', 'kind': 'op', 'op': 'Concat'},
    'concat_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Convolutions
    'conv_1': {'type': 'Convolution', 'kind': 'op', 'op': 'Conv2D', 'layout': 'NHWC'},
    'conv_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_1_b': {'value': None, 'shape': None, 'kind': 'data'},
    'const_conv_1_w': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None},
    'const_conv_1_b': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None},
    'conv_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_2': {'type': 'Convolution', 'kind': 'op', 'op': 'Conv2D', 'layout': 'NHWC'},
    'conv_2_w': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_2_b': {'value': None, 'shape': None, 'kind': 'data'},
    'const_conv_2_w': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None},
    'const_conv_2_b': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None},
    'conv_2_data': {'value': None, 'shape': None, 'kind': 'data'},
    # MatMul
    'fc_1': {'type': 'MatMul', 'kind': 'op', 'layout': 'NHWC'},
    'fc_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'fc_1_b': {'value': None, 'shape': None, 'kind': 'data'},
    'const_fc_1_w': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None},
    'const_fc_1_b': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None},
    'fc_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Placeholders
    'placeholder_2': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_3': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_3_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'op_output': {'kind': 'op', 'op': 'Result'},
    'op_output_1': {'kind': 'op', 'op': 'Result'},
    'op_output_2': {'kind': 'op', 'op': 'Result'}
}


# Unit tests for fuse_mul
class FuseMulTests(unittest.TestCase):
    # Mul(array)->Conv(w+b)
    def test_fuse_mul_to_conv_1(self):
        # Placeholder->Mul->Conv
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'conv_1'),
                             ('const_conv_1_w', 'conv_1_w'),
                             ('const_conv_1_b', 'conv_1_b'),
                             ('conv_1_w', 'conv_1'),
                             ('conv_1_b', 'conv_1'),
                             ('conv_1', 'conv_1_data'),
                             ('conv_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'const_mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'const_conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96))},
                             'conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96)),
                                          'output_channel_dim': 3, 'input_channel_dim': 2,
                                          'dims_number': 4},
                             'const_conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                             'conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                             'conv_1_data': {}
                             })
        ref_weights = np.ones((11, 11, 3, 96)) * np.reshape(np.array([1, 2, 3]), (3, 1))

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'conv_1'),
                                 ('const_conv_1_w', 'conv_1_w'),
                                 ('const_conv_1_b', 'conv_1_b'),
                                 ('conv_1_w', 'conv_1'),
                                 ('conv_1_b', 'conv_1'),
                                 ('conv_1', 'conv_1_data'),
                                 ('conv_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'const_conv_1_w': {'shape': ref_weights.shape, 'value': ref_weights},
                                 'conv_1_w': {'shape': ref_weights.shape, 'value': ref_weights,
                                              'output_channel_dim': 3, 'input_channel_dim': 2,
                                              'dims_number': 4},
                                 'const_conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_1_data': {}
                                 })

        _fuse_mul(graph, Node(graph, 'mul_1'), [Node(graph, 'conv_1')], backward=False)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1')
        self.assertTrue(flag, resp)

    # Mul(scalar)->Conv(w+b)
    def test_fuse_mul_to_conv_2(self):
        # Placeholder->Mul->Conv
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'conv_1'),
                             ('const_conv_1_w', 'conv_1_w'),
                             ('const_conv_1_b', 'conv_1_b'),
                             ('conv_1_w', 'conv_1'),
                             ('conv_1_b', 'conv_1'),
                             ('conv_1', 'conv_1_data'),
                             ('conv_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'const_mul_1_w': {'shape': np.array([1]), 'value': 6},
                             'mul_1_w': {'shape': np.array([1]), 'value': 6},
                             'const_conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96))},
                             'conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96)),
                                          'output_channel_dim': 3, 'input_channel_dim': 2,
                                          'dims_number': 4},
                             'const_conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                             'conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                             'conv_1_data': {}
                             })
        ref_weights = np.ones((11, 11, 3, 96)) * np.reshape(np.array([6, 6, 6]), (3, 1))

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'conv_1'),
                                 ('const_conv_1_w', 'conv_1_w'),
                                 ('const_conv_1_b', 'conv_1_b'),
                                 ('conv_1_w', 'conv_1'),
                                 ('conv_1_b', 'conv_1'),
                                 ('conv_1', 'conv_1_data'),
                                 ('conv_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'const_conv_1_w': {'shape': ref_weights.shape, 'value': ref_weights},
                                 'conv_1_w': {'shape': ref_weights.shape, 'value': ref_weights,
                                              'output_channel_dim': 3, 'input_channel_dim': 2,
                                              'dims_number': 4},
                                 'const_conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_1_data': {}
                                 })

        _fuse_mul(graph, Node(graph, 'mul_1'), [Node(graph, 'conv_1')], backward=False)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1')
        self.assertTrue(flag, resp)

    # Conv(w+b)->Mul(array)
    def test_fuse_mul_to_conv_3(self):
        # Placeholder->Conv->Mul
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'conv_1'),
                             ('const_conv_1_w', 'conv_1_w'),
                             ('const_conv_1_b', 'conv_1_b'),
                             ('conv_1_w', 'conv_1'),
                             ('conv_1_b', 'conv_1'),
                             ('conv_1', 'conv_1_data'),
                             ('conv_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'const_conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96))},
                             'conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96)),
                                          'output_channel_dim': 3, 'input_channel_dim': 2,
                                          'dims_number': 4},
                             'const_conv_1_b': {'shape': np.array([96]), 'value': np.ones(96)},
                             'conv_1_b': {'shape': np.array([96]), 'value': np.ones(96)},
                             'conv_1_data': {'shape': np.array([1, 55, 55, 96])},
                             'mul_1_data': {'shape': np.array([1, 55, 55, 96])},
                             'const_mul_1_w': {'shape': np.array([96]), 'value': np.array([x for x in range(96)])},
                             'mul_1_w': {'shape': np.array([96]), 'value': np.array([x for x in range(96)])},
                             })
        ref_weights = np.ones((11, 11, 3, 96)) * np.reshape(np.array([x for x in range(96)]), 96)
        ref_biases = np.ones(96) * np.array([x for x in range(96)])

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'conv_1'),
                                 ('const_conv_1_w', 'conv_1_w'),
                                 ('const_conv_1_b', 'conv_1_b'),
                                 ('conv_1_w', 'conv_1'),
                                 ('conv_1_b', 'conv_1'),
                                 ('conv_1', 'conv_1_data'),
                                 ('conv_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'const_conv_1_w': {'shape': ref_weights.shape, 'value': ref_weights},
                                 'conv_1_w': {'shape': ref_weights.shape, 'value': ref_weights,
                                              'output_channel_dim': 3, 'input_channel_dim': 2,
                                              'dims_number': 4},
                                 'const_conv_1_b': {'shape': ref_biases.shape, 'value': ref_biases},
                                 'conv_1_b': {'shape': ref_biases.shape, 'value': ref_biases},
                                 'conv_1_data': {'shape': np.array([1, 55, 55, 96])}
                                 })

        _fuse_mul(graph, Node(graph, 'mul_1'), [Node(graph, 'conv_1')], backward=True)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1', 'placeholder_1')
        self.assertTrue(flag, resp)

    # Conv(w)->Mul(scalar)
    def test_fuse_mul_to_conv_4(self):
        # Placeholder->Conv->Mul
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'conv_1'),
                             ('const_conv_1_w', 'conv_1_w'),
                             ('const_conv_1_b', 'conv_1_b'),
                             ('conv_1_w', 'conv_1'),
                             ('conv_1_b', 'conv_1'),
                             ('conv_1', 'conv_1_data'),
                             ('conv_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'const_conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96))},
                             'conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96)),
                                          'output_channel_dim': 3, 'input_channel_dim': 2,
                                          'dims_number': 4},
                             'const_conv_1_b': {'shape': np.array([96]), 'value': np.ones(96)},
                             'conv_1_b': {'shape': np.array([96]), 'value': np.ones(96)},
                             'conv_1_data': {'shape': np.array([1, 55, 55, 96])},
                             'mul_1_data': {'shape': np.array([1, 55, 55, 96])},
                             'const_mul_1_w': {'shape': np.array([1]), 'value': 6},
                             'mul_1_w': {'shape': np.array([1]), 'value': 6},
                             })
        ref_weights = np.ones((11, 11, 3, 96)) * np.array([6])
        ref_biases = np.ones(96) * np.array([6])

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'conv_1'),
                                 ('const_conv_1_w', 'conv_1_w'),
                                 ('const_conv_1_b', 'conv_1_b'),
                                 ('conv_1_w', 'conv_1'),
                                 ('conv_1_b', 'conv_1'),
                                 ('conv_1', 'conv_1_data'),
                                 ('conv_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'const_conv_1_w': {'shape': ref_weights.shape, 'value': ref_weights},
                                 'conv_1_w': {'shape': ref_weights.shape, 'value': ref_weights,
                                              'output_channel_dim': 3, 'input_channel_dim': 2,
                                              'dims_number': 4},
                                 'const_conv_1_b': {'shape': ref_biases.shape, 'value': ref_biases},
                                 'conv_1_b': {'shape': ref_biases.shape, 'value': ref_biases},
                                 'conv_1_data': {'shape': np.array([1, 55, 55, 96])}
                                 })

        _fuse_mul(graph, Node(graph, 'mul_1'), [Node(graph, 'conv_1')], backward=True)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1')
        self.assertTrue(flag, resp)

    # Op0-+->Op1--+----+-->Concat     Op0-+->Op1--+--+-->Concat
    #  |  |       |    |               |  |       |  |
    #  |  +->Op2--+    |          =>   |  +->Op2--+  |
    #  +---->Mul->Conv-+               +---->Conv----+
    def test_fuse_mul_to_conv_5(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'conv_1'),
                             ('const_conv_1_w', 'conv_1_w'),
                             ('const_conv_1_b', 'conv_1_b'),
                             ('conv_1_w', 'conv_1'),
                             ('conv_1_b', 'conv_1'),
                             ('conv_1', 'conv_1_data'),
                             ('placeholder_1_data', 'placeholder_2'),
                             ('placeholder_2', 'placeholder_2_data'),
                             ('placeholder_1_data', 'placeholder_3'),
                             ('placeholder_3', 'placeholder_3_data'),
                             ('placeholder_2_data', 'concat_1'),
                             ('placeholder_3_data', 'concat_1'),
                             ('conv_1_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('concat_1_data', 'op_output')

                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'const_mul_1_w': {'shape': np.array([1]), 'value': 6},
                             'mul_1_w': {'shape': np.array([1]), 'value': 6},
                             'const_conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96))},
                             'conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96)),
                                          'output_channel_dim': 3, 'input_channel_dim': 2,
                                          'dims_number': 4},
                             'const_conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                             'conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                             'concat_1_data': {}
                             })
        ref_weights = np.ones((11, 11, 3, 96)) * np.reshape(np.array([6, 6, 6]), (3, 1))

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'conv_1'),
                                 ('const_conv_1_w', 'conv_1_w'),
                                 ('const_conv_1_b', 'conv_1_b'),
                                 ('conv_1_w', 'conv_1'),
                                 ('conv_1_b', 'conv_1'),
                                 ('conv_1', 'conv_1_data'),
                                 ('placeholder_1_data', 'placeholder_2'),
                                 ('placeholder_2', 'placeholder_2_data'),
                                 ('placeholder_1_data', 'placeholder_3'),
                                 ('placeholder_3', 'placeholder_3_data'),
                                 ('placeholder_2_data', 'concat_1'),
                                 ('placeholder_3_data', 'concat_1'),
                                 ('conv_1_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('concat_1_data', 'op_output'),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'const_conv_1_w': {'shape': ref_weights.shape, 'value': ref_weights},
                                 'conv_1_w': {'shape': ref_weights.shape, 'value': ref_weights,
                                              'output_channel_dim': 3,
                                              'input_channel_dim': 2, 'dims_number': 4},
                                 'const_conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_1_data': {},
                                 'placeholder_2_data': {},
                                 'placeholder_3_data': {},
                                 })

        _fuse_mul(graph, Node(graph, 'mul_1'), [Node(graph, 'conv_1')], backward=False)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data')
        self.assertTrue(flag, resp)

    def test_fuse_mul_to_conv_5_nparray(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'conv_1'),
                             ('const_conv_1_w', 'conv_1_w'),
                             ('const_conv_1_b', 'conv_1_b'),
                             ('conv_1_w', 'conv_1'),
                             ('conv_1_b', 'conv_1'),
                             ('conv_1', 'conv_1_data'),
                             ('placeholder_1_data', 'placeholder_2'),
                             ('placeholder_2', 'placeholder_2_data'),
                             ('placeholder_1_data', 'placeholder_3'),
                             ('placeholder_3', 'placeholder_3_data'),
                             ('placeholder_2_data', 'concat_1'),
                             ('placeholder_3_data', 'concat_1'),
                             ('conv_1_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('concat_1_data', 'op_output'),

                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'const_mul_1_w': {'shape': np.array([1]), 'value': np.array([6])},
                             'mul_1_w': {'shape': np.array([1]), 'value': np.array([6])},
                             'const_conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96))},
                             'conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96)),
                                          'output_channel_dim': 3, 'input_channel_dim': 2,
                                          'dims_number': 4},
                             'const_conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                             'conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                             'concat_1_data': {}
                             })
        ref_weights = np.ones((11, 11, 3, 96)) * np.reshape(np.array([6, 6, 6]), (3, 1))

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'conv_1'),
                                 ('const_conv_1_w', 'conv_1_w'),
                                 ('const_conv_1_b', 'conv_1_b'),
                                 ('conv_1_w', 'conv_1'),
                                 ('conv_1_b', 'conv_1'),
                                 ('conv_1', 'conv_1_data'),
                                 ('placeholder_1_data', 'placeholder_2'),
                                 ('placeholder_2', 'placeholder_2_data'),
                                 ('placeholder_1_data', 'placeholder_3'),
                                 ('placeholder_3', 'placeholder_3_data'),
                                 ('placeholder_2_data', 'concat_1'),
                                 ('placeholder_3_data', 'concat_1'),
                                 ('conv_1_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('concat_1_data', 'op_output'),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'const_conv_1_w': {'shape': ref_weights.shape, 'value': ref_weights},
                                 'conv_1_w': {'shape': ref_weights.shape, 'value': ref_weights,
                                              'output_channel_dim': 3,
                                              'input_channel_dim': 2, 'dims_number': 4},
                                 'const_conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_1_data': {},
                                 'placeholder_2_data': {},
                                 'placeholder_3_data': {},
                                 })

        _fuse_mul(graph, Node(graph, 'mul_1'), [Node(graph, 'conv_1')], backward=False)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data')
        self.assertTrue(flag, resp)

    # Op->Mul(array)-+->Conv(w+b)--+->Concat     Op-+->Conv1-+-->Concat
    #                |             |         =>     |        |
    #                +-->Conv(w+b)-+                +->Conv2-+
    def test_fuse_mul_to_convolutions_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'conv_1'),
                             ('const_conv_1_w', 'conv_1_w'),
                             ('const_conv_1_b', 'conv_1_b'),
                             ('conv_1_w', 'conv_1'),
                             ('conv_1_b', 'conv_1'),
                             ('conv_1', 'conv_1_data'),
                             ('mul_1_data', 'conv_2'),
                             ('const_conv_2_w', 'conv_2_w'),
                             ('const_conv_2_b', 'conv_2_b'),
                             ('conv_2_w', 'conv_2'),
                             ('conv_2_b', 'conv_2'),
                             ('conv_2', 'conv_2_data'),
                             ('conv_1_data', 'concat_1'),
                             ('conv_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('concat_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'const_mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'const_conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96))},
                             'conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96)),
                                          'output_channel_dim': 3, 'input_channel_dim': 2,
                                          'dims_number': 4},
                             'const_conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                             'conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                             'conv_1_data': {'shape': np.array([1, 55, 55, 96])},
                             'const_conv_2_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96))},
                             'conv_2_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96)),
                                          'output_channel_dim': 3, 'input_channel_dim': 2,
                                          'dims_number': 4},
                             'const_conv_2_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                             'conv_2_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                             'conv_2_data': {'shape': np.array([1, 55, 55, 96])},
                             'concat_1_data': {}
                             })
        ref_weights = np.ones((11, 11, 3, 96)) * np.reshape(np.array([1, 2, 3]), (3, 1))

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'conv_1'),
                                 ('const_conv_1_w', 'conv_1_w'),
                                 ('const_conv_1_b', 'conv_1_b'),
                                 ('conv_1_w', 'conv_1'),
                                 ('conv_1_b', 'conv_1'),
                                 ('conv_1', 'conv_1_data'),
                                 ('placeholder_1_data', 'conv_2'),
                                 ('const_conv_2_w', 'conv_2_w'),
                                 ('const_conv_2_b', 'conv_2_b'),
                                 ('conv_2_w', 'conv_2'),
                                 ('conv_2_b', 'conv_2'),
                                 ('conv_2', 'conv_2_data'),
                                 ('conv_1_data', 'concat_1'),
                                 ('conv_2_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('concat_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'const_conv_1_w': {'shape': ref_weights.shape, 'value': ref_weights},
                                 'conv_1_w': {'shape': ref_weights.shape, 'value': ref_weights,
                                              'output_channel_dim': 3, 'input_channel_dim': 2,
                                              'dims_number': 4},
                                 'const_conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_1_data': {'shape': np.array([1, 55, 55, 96])},
                                 'const_conv_2_w': {'shape': ref_weights.shape, 'value': ref_weights},
                                 'conv_2_w': {'shape': ref_weights.shape, 'value': ref_weights,
                                              'output_channel_dim': 3, 'input_channel_dim': 2,
                                              'dims_number': 4},
                                 'const_conv_2_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_2_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_2_data': {'shape': np.array([1, 55, 55, 96])},
                                 })

        _fuse_mul(graph, Node(graph, 'mul_1'), [Node(graph, 'conv_1'), Node(graph, 'conv_2')], backward=False)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1')
        self.assertTrue(flag, resp)

    # Mul(array)->FC(w+b)
    def test_fuse_mul_to_fc_1(self):
        # Placeholder->Mul->FC
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'fc_1'),
                             ('const_fc_1_w', 'fc_1_w'),
                             ('const_fc_1_b', 'fc_1_b'),
                             ('fc_1_w', 'fc_1'),
                             ('fc_1_b', 'fc_1'),
                             ('fc_1', 'fc_1_data'),
                             ('fc_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 2048])},
                             'mul_1_data': {'shape': np.array([1, 2048])},
                             'const_mul_1_w': {'shape': np.array([2048]), 'value': np.array([x for x in range(2048)])},
                             'mul_1_w': {'shape': np.array([2048]), 'value': np.array([x for x in range(2048)])},
                             'const_fc_1_w': {'shape': np.array([10260, 2048]), 'value': np.ones((10260, 2048))},
                             'fc_1_w': {'shape': np.array([10260, 2048]), 'value': np.ones((10260, 2048)),
                                        'output_channel_dim': 0, 'input_channel_dim': 1,
                                        'dims_number': 2},
                             'const_fc_1_b': {'shape': np.array([10260]), 'value': np.ones(10260)},
                             'fc_1_b': {'shape': np.array([10260]), 'value': np.ones(10260)},
                             'fc_1_data': {'shape': np.array([1, 10260])},
                             })
        ref_weights = np.ones((10260, 2048)) * np.array([x for x in range(2048)])

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'fc_1'),
                                 ('const_fc_1_w', 'fc_1_w'),
                                 ('const_fc_1_b', 'fc_1_b'),
                                 ('fc_1_w', 'fc_1'),
                                 ('fc_1_b', 'fc_1'),
                                 ('fc_1', 'fc_1_data'),
                                 ('fc_1_data', 'op_output')

                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 2048])},
                                 'const_fc_1_w': {'shape': ref_weights.shape, 'value': ref_weights},
                                 'fc_1_w': {'shape': ref_weights.shape, 'value': ref_weights,
                                            'output_channel_dim': 0, 'input_channel_dim': 1,
                                            'dims_number': 2},
                                 'const_fc_1_b': {'shape': np.array([10260]), 'value': np.ones(10260)},
                                 'fc_1_b': {'shape': np.array([10260]), 'value': np.ones(10260)},
                                 'fc_1_data': {'shape': np.array([1, 10260])},
                                 })

        _fuse_mul(graph, Node(graph, 'mul_1'), [Node(graph, 'fc_1')], backward=False)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1')
        self.assertTrue(flag, resp)

    # Mul(scalar)->Conv(w+b) can_be_fused = False
    def test_fuse_mul_to_conv_6(self):
        # Placeholder->Mul->Conv
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'conv_1'),
                             ('const_conv_1_w', 'conv_1_w'),
                             ('const_conv_1_b', 'conv_1_b'),
                             ('conv_1_w', 'conv_1'),
                             ('conv_1_b', 'conv_1'),
                             ('conv_1', 'conv_1_data'),
                             ('conv_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'const_mul_1_w': {'shape': np.array([]), 'value': np.array(6)},
                             'mul_1_w': {'shape': np.array([]), 'value': np.array(6)},
                             'conv_1': {'can_be_fused': False},
                             'const_conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96))},
                             'conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96)),
                                          'output_channel_dim': 3, 'input_channel_dim': 2,
                                          'dims_number': 4},
                             'const_conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                             'conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                             'conv_1_data': {}
                             })

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('const_mul_1_w', 'mul_1_w'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'mul_1_data'),
                                 ('mul_1_data', 'conv_1'),
                                 ('const_conv_1_w', 'conv_1_w'),
                                 ('const_conv_1_b', 'conv_1_b'),
                                 ('conv_1_w', 'conv_1'),
                                 ('conv_1_b', 'conv_1'),
                                 ('conv_1', 'conv_1_data'),
                                 ('conv_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'const_mul_1_w': {'shape': np.array([]), 'value': np.array(6)},
                                 'mul_1_w': {'shape': np.array([]), 'value': np.array(6)},
                                 'conv_1': {'can_be_fused': False},
                                 'const_conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96))},
                                 'conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96)),
                                              'output_channel_dim': 3, 'input_channel_dim': 2,
                                              'dims_number': 4},
                                 'const_conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_1_data': {}
                                 })

        _fuse_mul(graph, Node(graph, 'mul_1'), [Node(graph, 'conv_1')], backward=False)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1')
        self.assertTrue(flag, resp)

    # Mul(array)->DWConv(w+b)
    def test_fuse_mul_to_dwconv_1(self):
        # Placeholder->Mul->Conv
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'conv_1'),
                             ('const_conv_1_w', 'conv_1_w'),
                             ('conv_1_w', 'conv_1'),
                             ('conv_1', 'conv_1_data'),
                             ('conv_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 112, 112, 6])},
                             'mul_1_data': {'shape': np.array([1, 112, 112, 6])},
                             'const_mul_1_w': {'shape': np.array([6]), 'value': np.array([1, 2, 3, 4, 5, 6])},
                             'mul_1_w': {'shape': np.array([6]), 'value': np.array([1, 2, 3, 4, 5, 6])},
                             'const_conv_1_w': {'shape': np.array([3, 3, 6, 1]), 'value': np.ones((3, 3, 6, 1))},
                             'conv_1_w': {'shape': np.array([3, 3, 6, 1]), 'value': np.ones((3, 3, 6, 1)),
                                          'output_channel_dim': 2, 'input_channel_dim': 2,
                                          'dims_number': 4},
                             'conv_1_data': {}
                             })
        ref_weights = np.ones((3, 3, 6, 1)) * np.reshape(np.array([1, 2, 3, 4, 5, 6]), (6, 1))

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'conv_1'),
                                 ('const_conv_1_w', 'conv_1_w'),
                                 ('conv_1_w', 'conv_1'),
                                 ('conv_1', 'conv_1_data'),
                                 ('conv_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 112, 112, 6])},
                                 'const_conv_1_w': {'shape': ref_weights.shape, 'value': ref_weights},
                                 'conv_1_w': {'shape': ref_weights.shape, 'value': ref_weights,
                                              'output_channel_dim': 2, 'input_channel_dim': 2,
                                              'dims_number': 4},
                                 'conv_1_data': {}
                                 })

        _fuse_mul(graph, Node(graph, 'mul_1'), [Node(graph, 'conv_1')], backward=False)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1')
        self.assertTrue(flag, resp)

    # DWConv(w)->Mul(scalar)
    def test_fuse_mul_to_dwconv_2(self):
        # Placeholder->Conv->Mul
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'conv_1'),
                             ('const_conv_1_w', 'conv_1_w'),
                             ('conv_1_w', 'conv_1'),
                             ('conv_1', 'conv_1_data'),
                             ('conv_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 112, 112, 6])},
                             'mul_1_data': {'shape': np.array([1, 112, 112, 6])},
                             'const_mul_1_w': {'shape': np.array([6]), 'value': np.array([1, 2, 3, 4, 5, 6])},
                             'mul_1_w': {'shape': np.array([6]), 'value': np.array([1, 2, 3, 4, 5, 6])},
                             'const_conv_1_w': {'shape': np.array([3, 3, 6, 1]), 'value': np.ones((3, 3, 6, 1))},
                             'conv_1_w': {'shape': np.array([3, 3, 6, 1]), 'value': np.ones((3, 3, 6, 1)),
                                          'output_channel_dim': 2, 'input_channel_dim': 2,
                                          'dims_number': 4},
                             'conv_1_data': {}
                             })

        ref_weights = np.ones((3, 3, 6, 1)) * np.reshape(np.array([1, 2, 3, 4, 5, 6]), (6, 1))

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'conv_1'),
                                 ('const_conv_1_w', 'conv_1_w'),
                                 ('conv_1_w', 'conv_1'),
                                 ('conv_1', 'conv_1_data'),
                                 ('conv_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 112, 112, 6])},
                                 'const_conv_1_w': {'shape': ref_weights.shape, 'value': ref_weights},
                                 'conv_1_w': {'shape': ref_weights.shape, 'value': ref_weights,
                                              'output_channel_dim': 2, 'input_channel_dim': 2,
                                              'dims_number': 4},
                                 })

        _fuse_mul(graph, Node(graph, 'mul_1'), [Node(graph, 'conv_1')], backward=True)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1', 'placeholder_1')
        self.assertTrue(flag, resp)


# Unit tests for fuse_add
class FuseAddTests(unittest.TestCase):
    # Add(array)->FC(w+b)
    def test_fuse_add_to_fc_1(self):
        # Placeholder->Add->FC
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'add_1'),
                             ('const_add_1_w', 'add_1_w'),
                             ('add_1_w', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'fc_1'),
                             ('const_fc_1_w', 'fc_1_w'),
                             ('const_fc_1_b', 'fc_1_b'),
                             ('fc_1_w', 'fc_1'),
                             ('fc_1_b', 'fc_1'),
                             ('fc_1', 'fc_1_data'),
                             ('fc_1_data', 'op_output')

                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 2048])},
                             'add_1_data': {'shape': np.array([1, 2048])},
                             'const_add_1_w': {'shape': np.array([2048]), 'value': np.array([x for x in range(2048)])},
                             'add_1_w': {'shape': np.array([2048]), 'value': np.array([x for x in range(2048)])},
                             'const_fc_1_w': {'shape': np.array([10260, 2048]), 'value': np.ones((10260, 2048))},
                             'fc_1_w': {'shape': np.array([10260, 2048]), 'value': np.ones((10260, 2048)),
                                        'output_channel_dim': 0, 'input_channel_dim': 1,
                                        'dims_number': 2},
                             'const_fc_1_b': {'shape': np.array([10260]), 'value': np.ones(10260)},
                             'fc_1_b': {'shape': np.array([10260]), 'value': np.ones(10260)},
                             'fc_1_data': {'shape': np.array([1, 10260])},
                             })
        ref_weights = np.ones((10260, 2048))
        ref_biases = np.ones(10260) + np.dot(np.ones((10260, 2048)), np.array([x for x in range(2048)]))

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'fc_1'),
                                 ('const_fc_1_w', 'fc_1_w'),
                                 ('const_fc_1_b', 'fc_1_b'),
                                 ('fc_1_w', 'fc_1'),
                                 ('fc_1_b', 'fc_1'),
                                 ('fc_1', 'fc_1_data'),
                                 ('fc_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 2048])},
                                 'const_fc_1_w': {'shape': ref_weights.shape, 'value': ref_weights},
                                 'fc_1_w': {'shape': ref_weights.shape, 'value': ref_weights,
                                            'output_channel_dim': 0, 'input_channel_dim': 1,
                                            'dims_number': 2},
                                 'const_fc_1_b': {'shape': ref_biases.shape, 'value': ref_biases},
                                 'fc_1_b': {'shape': ref_biases.shape, 'value': ref_biases},
                                 'fc_1_data': {'shape': np.array([1, 10260])},
                                 })

        _fuse_add(graph, Node(graph, 'add_1'), [Node(graph, 'fc_1')], backward=False)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1')
        self.assertTrue(flag, resp)

    # FC(w)->Add(array)
    def test_fuse_add_to_fc_2(self):
        # Placeholder->FC->Add
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'fc_1'),
                             ('const_fc_1_w', 'fc_1_w'),
                             ('fc_1_w', 'fc_1'),
                             ('fc_1', 'fc_1_data'),
                             ('fc_1_data', 'add_1'),
                             ('const_add_1_w', 'add_1_w'),
                             ('add_1_w', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'op_output_1')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 2048])},
                             'add_1_data': {'shape': np.array([1, 10260])},
                             'const_add_1_w': {'shape': np.array([10260]), 'value': np.array([x for x in range(10260)])},
                             'add_1_w': {'shape': np.array([10260]), 'value': np.array([x for x in range(10260)]),
                                         'data_type': None},
                             'const_fc_1_w': {'shape': np.array([10260, 2048]), 'value': np.ones((10260, 2048))},
                             'fc_1_w': {'shape': np.array([10260, 2048]), 'value': np.ones((10260, 2048)),
                                        'output_channel_dim': 0, 'input_channel_dim': 1,
                                        'dims_number': 2, 'data_type': None},
                             'fc_1_data': {'shape': np.array([1, 10260])},
                             })

        ref_weights = np.ones((10260, 2048))
        ref_biases = np.array([x for x in range(10260)])

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'fc_1'),
                                 ('const_fc_1_w', 'fc_1_w'),
                                 ('const_fc_1_b', 'fc_1_b'),
                                 ('fc_1_w', 'fc_1'),
                                 ('fc_1_b', 'fc_1'),
                                 ('fc_1', 'fc_1_data'),
                                 ('fc_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 2048])},
                                 'const_fc_1_w': {'shape': ref_weights.shape, 'value': ref_weights},
                                 'fc_1_w': {'shape': ref_weights.shape, 'value': ref_weights,
                                            'output_channel_dim': 0, 'input_channel_dim': 1,
                                            'dims_number': 2},
                                 'const_fc_1_b': {'shape': ref_biases.shape, 'value': ref_biases},
                                 'fc_1_b': {'shape': ref_biases.shape, 'value': ref_biases},
                                 'fc_1_data': {'shape': np.array([1, 10260])},
                                 })

        _fuse_add(graph, Node(graph, 'add_1'), [Node(graph, 'fc_1')], backward=True)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1')
        self.assertTrue(flag, resp)

    # FC(w)->Add(scalar)
    def test_fuse_add_to_fc_3(self):
        # Placeholder->FC->Add
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'fc_1'),
                             ('const_fc_1_w', 'fc_1_w'),
                             ('fc_1_w', 'fc_1'),
                             ('fc_1', 'fc_1_data'),
                             ('fc_1_data', 'add_1'),
                             ('const_add_1_w', 'add_1_w'),
                             ('add_1_w', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 2048])},
                             'add_1_data': {'shape': np.array([1, 10260])},
                             'const_add_1_w': {'shape': np.array([1]), 'value': 6, 'data_type': None},
                             'add_1_w': {'shape': np.array([1]), 'value': 6, 'data_type': None},
                             'const_fc_1_w': {'shape': np.array([10260, 2048]), 'value': np.ones((10260, 2048))},
                             'fc_1_w': {'shape': np.array([10260, 2048]), 'value': np.ones((10260, 2048)),
                                        'output_channel_dim': 0, 'input_channel_dim': 1,
                                        'dims_number': 2, 'data_type': None},
                             'fc_1_data': {'shape': np.array([1, 10260])},
                             })

        ref_weights = np.ones((10260, 2048))
        ref_biases = np.full([10260], 6)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'fc_1'),
                                 ('const_fc_1_w', 'fc_1_w'),
                                 ('const_fc_1_b', 'fc_1_b'),
                                 ('fc_1_w', 'fc_1'),
                                 ('fc_1_b', 'fc_1'),
                                 ('fc_1', 'fc_1_data'),
                                 ('fc_1_data', 'op_output')

                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 2048])},
                                 'const_fc_1_w': {'shape': ref_weights.shape, 'value': ref_weights},
                                 'fc_1_w': {'shape': ref_weights.shape, 'value': ref_weights,
                                            'output_channel_dim': 0, 'input_channel_dim': 1,
                                            'dims_number': 2},
                                 'const_fc_1_b': {'shape': ref_biases.shape, 'value': ref_biases},
                                 'fc_1_b': {'shape': ref_biases.shape, 'value': ref_biases},
                                 'fc_1_data': {'shape': np.array([1, 10260])},
                                 })

        _fuse_add(graph, Node(graph, 'add_1'), [Node(graph, 'fc_1')], backward=True)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1')
        self.assertTrue(flag, resp)

    # Add(array)->FC(w+b) can_be_fused = False
    def test_fuse_add_to_fc_4(self):
        # Placeholder->Add->FC
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'add_1'),
                             ('const_add_1_w', 'add_1_w'),
                             ('add_1_w', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'fc_1'),
                             ('const_fc_1_w', 'fc_1_w'),
                             ('const_fc_1_b', 'fc_1_b'),
                             ('fc_1_w', 'fc_1'),
                             ('fc_1_b', 'fc_1'),
                             ('fc_1', 'fc_1_data'),
                             ('fc_1_data', 'op_output')

                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 2048])},
                             'add_1_data': {'shape': np.array([1, 2048])},
                             'const_add_1_w': {'shape': np.array([2048]), 'value': np.array([x for x in range(2048)])},
                             'add_1_w': {'shape': np.array([2048]), 'value': np.array([x for x in range(2048)])},
                             'fc_1': {'can_be_fused': False},
                             'const_fc_1_w': {'shape': np.array([10260, 2048]), 'value': np.ones((10260, 2048))},
                             'fc_1_w': {'shape': np.array([10260, 2048]), 'value': np.ones((10260, 2048)),
                                        'output_channel_dim': 0, 'input_channel_dim': 1,
                                        'dims_number': 2},
                             'const_fc_1_b': {'shape': np.array([10260]), 'value': np.ones(10260)},
                             'fc_1_b': {'shape': np.array([10260]), 'value': np.ones(10260)},
                             'fc_1_data': {'shape': np.array([1, 10260])},
                             })

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'add_1'),
                                 ('const_add_1_w', 'add_1_w'),
                                 ('add_1_w', 'add_1'),
                                 ('add_1', 'add_1_data'),
                                 ('add_1_data', 'fc_1'),
                                 ('const_fc_1_w', 'fc_1_w'),
                                 ('const_fc_1_b', 'fc_1_b'),
                                 ('fc_1_w', 'fc_1'),
                                 ('fc_1_b', 'fc_1'),
                                 ('fc_1', 'fc_1_data'),
                                 ('fc_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 2048])},
                                 'add_1_data': {'shape': np.array([1, 2048])},
                                 'const_add_1_w': {'shape': np.array([2048]), 'value': np.array([x for x in range(2048)])},
                                 'add_1_w': {'shape': np.array([2048]), 'value': np.array([x for x in range(2048)])},
                                 'fc_1': {'can_be_fused': False},
                                 'const_fc_1_w': {'shape': np.array([10260, 2048]), 'value': np.ones((10260, 2048))},
                                 'fc_1_w': {'shape': np.array([10260, 2048]), 'value': np.ones((10260, 2048)),
                                            'output_channel_dim': 0, 'input_channel_dim': 1,
                                            'dims_number': 2},
                                 'const_fc_1_b': {'shape': np.array([10260]), 'value': np.ones(10260)},
                                 'fc_1_b': {'shape': np.array([10260]), 'value': np.ones(10260)},
                                 'fc_1_data': {'shape': np.array([1, 10260])},
                                 })

        _fuse_add(graph, Node(graph, 'add_1'), [Node(graph, 'fc_1')], backward=False)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1')
        self.assertTrue(flag, resp)


# Unit tests for fuse_linear_ops
class FuseLinOpsTests(unittest.TestCase):
    # Op->Mul(array)-+->Conv(w+b)->Add-+->Concat     Op-+->Conv1-+-->Concat
    #                |                 |         =>     |        |
    #                +-->Conv(w+b)-----+                +->Conv2-+
    def test_fuse_lin_ops_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'conv_1'),
                             ('const_conv_1_w', 'conv_1_w'),
                             ('const_conv_1_b', 'conv_1_b'),
                             ('conv_1_w', 'conv_1'),
                             ('conv_1_b', 'conv_1'),
                             ('conv_1', 'conv_1_data'),
                             ('mul_1_data', 'conv_2'),
                             ('const_conv_2_w', 'conv_2_w'),
                             ('const_conv_2_b', 'conv_2_b'),
                             ('conv_2_w', 'conv_2'),
                             ('conv_2_b', 'conv_2'),
                             ('conv_2', 'conv_2_data'),
                             ('conv_1_data', 'concat_1'),
                             ('conv_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('concat_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'const_mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'const_conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96))},
                             'conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96)),
                                          'output_channel_dim': 3, 'input_channel_dim': 2,
                                          'dims_number': 4},
                             'const_conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                             'conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                             'conv_1_data': {'shape': np.array([1, 55, 55, 96])},
                             'const_conv_2_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96))},
                             'conv_2_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96)),
                                          'output_channel_dim': 3, 'input_channel_dim': 2,
                                          'dims_number': 4},
                             'const_conv_2_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                             'conv_2_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                             'conv_2_data': {'shape': np.array([1, 55, 55, 96])},
                             'concat_1_data': {}
                             })
        ref_weights = np.ones((11, 11, 3, 96)) * np.reshape(np.array([1, 2, 3]), (3, 1))

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'conv_1'),
                                 ('const_conv_1_w', 'conv_1_w'),
                                 ('const_conv_1_b', 'conv_1_b'),
                                 ('conv_1_w', 'conv_1'),
                                 ('conv_1_b', 'conv_1'),
                                 ('conv_1', 'conv_1_data'),
                                 ('placeholder_1_data', 'conv_2'),
                                 ('const_conv_2_w', 'conv_2_w'),
                                 ('const_conv_2_b', 'conv_2_b'),
                                 ('conv_2_w', 'conv_2'),
                                 ('conv_2_b', 'conv_2'),
                                 ('conv_2', 'conv_2_data'),
                                 ('conv_1_data', 'concat_1'),
                                 ('conv_2_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('concat_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'const_conv_1_w': {'shape': ref_weights.shape, 'value': ref_weights},
                                 'conv_1_w': {'shape': ref_weights.shape, 'value': ref_weights,
                                              'output_channel_dim': 3, 'input_channel_dim': 2,
                                              'dims_number': 4},
                                 'const_conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_1_data': {'shape': np.array([1, 55, 55, 96])},
                                 'const_conv_2_w': {'shape': ref_weights.shape, 'value': ref_weights},
                                 'conv_2_w': {'shape': ref_weights.shape, 'value': ref_weights,
                                              'output_channel_dim': 3, 'input_channel_dim': 2,
                                              'dims_number': 4},
                                 'const_conv_2_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_2_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_2_data': {'shape': np.array([1, 55, 55, 96])},
                                 })

        fuse_linear_ops(graph)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data')
        self.assertTrue(flag, resp)

    # Mul(array)->FC(w+b)
    def test_fuse_mul_to_fc_1(self):
        # Placeholder->Mul->FC
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'fc_1'),
                             ('const_fc_1_w', 'fc_1_w'),
                             ('const_fc_1_b', 'fc_1_b'),
                             ('fc_1_w', 'fc_1'),
                             ('fc_1_b', 'fc_1'),
                             ('fc_1', 'fc_1_data'),
                             ('fc_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 2048])},
                             'mul_1_data': {'shape': np.array([1, 2048])},
                             'const_mul_1_w': {'shape': np.array([2048]), 'value': np.array([x for x in range(2048)])},
                             'mul_1_w': {'shape': np.array([2048]), 'value': np.array([x for x in range(2048)])},
                             'const_fc_1_w': {'shape': np.array([10260, 2048]), 'value': np.ones((10260, 2048))},
                             'fc_1_w': {'shape': np.array([10260, 2048]), 'value': np.ones((10260, 2048)),
                                        'output_channel_dim': 0, 'input_channel_dim': 1,
                                        'dims_number': 2},
                             'const_fc_1_b': {'shape': np.array([10260]), 'value': np.ones(10260)},
                             'fc_1_b': {'shape': np.array([10260]), 'value': np.ones(10260)},
                             'fc_1_data': {'shape': np.array([1, 10260])},
                             })
        ref_weights = np.ones((10260, 2048)) * np.array([x for x in range(2048)])

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'fc_1'),
                                 ('const_fc_1_w', 'fc_1_w'),
                                 ('const_fc_1_b', 'fc_1_b'),
                                 ('fc_1_w', 'fc_1'),
                                 ('fc_1_b', 'fc_1'),
                                 ('fc_1', 'fc_1_data'),
                                 ('fc_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 2048])},
                                 'const_fc_1_w': {'shape': ref_weights.shape, 'value': ref_weights},
                                 'fc_1_w': {'shape': ref_weights.shape, 'value': ref_weights,
                                            'output_channel_dim': 0, 'input_channel_dim': 1,
                                            'dims_number': 2},
                                 'const_fc_1_b': {'shape': np.array([10260]), 'value': np.ones(10260)},
                                 'fc_1_b': {'shape': np.array([10260]), 'value': np.ones(10260)},
                                 'fc_1_data': {'shape': np.array([1, 10260])},
                                 })

        fuse_linear_ops(graph)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'fc_1_data')
        self.assertTrue(flag, resp)

    # FC(w)->Add(scalar)
    @unittest.skip("FC fusion is disabled")
    def test_fuse_add_to_fc_3(self):
        # Placeholder->FC->Add
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'fc_1'),
                             ('const_fc_1_w', 'fc_1_w'),
                             ('fc_1_w', 'fc_1'),
                             ('fc_1', 'fc_1_data'),
                             ('fc_1_data', 'add_1'),
                             ('const_add_1_w', 'add_1_w'),
                             ('add_1_w', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 2048])},
                             'add_1_data': {'shape': np.array([1, 10260])},
                             'const_add_1_w': {'shape': np.array([1]), 'value': np.array([6]), 'data_type': None},
                             'add_1_w': {'shape': np.array([1]), 'value': np.array([6]), 'data_type': None},
                             'const_fc_1_w': {'shape': np.array([10260, 2048]), 'value': np.ones((10260, 2048))},
                             'fc_1_w': {'shape': np.array([10260, 2048]), 'value': np.ones((10260, 2048)),
                                        'output_channel_dim': 0, 'input_channel_dim': 1,
                                        'dims_number': 2, 'data_type': None},
                             'fc_1_data': {'shape': np.array([1, 10260])},
                             })

        ref_weights = np.ones((10260, 2048))
        ref_biases = np.array([6 for x in range(10260)])

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'fc_1'),
                                 ('const_fc_1_w', 'fc_1_w'),
                                 ('const_fc_1_b', 'fc_1_b'),
                                 ('fc_1_w', 'fc_1'),
                                 ('fc_1_b', 'fc_1'),
                                 ('fc_1', 'fc_1_data'),
                                 ('fc_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 2048])},
                                 'const_fc_1_w': {'shape': ref_weights.shape, 'value': ref_weights},
                                 'fc_1_w': {'shape': ref_weights.shape, 'value': ref_weights,
                                            'output_channel_dim': 0, 'input_channel_dim': 1,
                                            'dims_number': 2},
                                 'const_fc_1_b': {'shape': ref_biases.shape, 'value': ref_biases},
                                 'fc_1_b': {'shape': ref_biases.shape, 'value': ref_biases},
                                 'fc_1_data': {'shape': np.array([1, 10260])},
                                 })

        fuse_linear_ops(graph)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1')
        self.assertTrue(flag, resp)

    #                 +-----------+
    #                 |           |           =>  Same
    # Placeholder--->Add->Mul-----+->Concat
    def test_fuse_lin_op_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1_data', 'conv_1'),
                             ('conv_1', 'conv_1_data'),
                             ('const_conv_1_w', 'conv_1_w'),
                             ('const_conv_1_b', 'conv_1_b'),
                             ('conv_1_w', 'conv_1'),
                             ('conv_1_b', 'conv_1'),
                             ('conv_1_data', 'add_1'),
                             ('const_add_1_w', 'add_1_w'),
                             ('add_1_w', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('concat_1', 'concat_1_data'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('add_1_data', 'concat_1'),
                             ('mul_1_data', 'concat_1'),
                             ('add_1_data', 'mul_1'),
                             ('concat_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'const_conv_1_w': {'shape': np.array([1, 1, 3, 3]), 'value': np.zeros((1, 1, 3, 3))},
                             'conv_1_w': {'shape': np.array([1, 1, 3, 3]), 'value': np.zeros((1, 1, 3, 3)),
                                          'output_channel_dim': 3, 'input_channel_dim': 2,
                                          'dims_number': 4},
                             'const_conv_1_b': {'shape': np.array([3]), 'value': np.zeros(3)},
                             'conv_1_b': {'shape': np.array([3]), 'value': np.zeros(3)},
                             'conv_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'const_mul_1_w': {'shape': np.array([1]), 'value': np.array([6])},
                             'mul_1_w': {'shape': np.array([1]), 'value': np.array([6])},
                             'const_add_1_w': {'shape': np.array([1]), 'value': np.array([1])},
                             'add_1_w': {'shape': np.array([1]), 'value': np.array([1])},
                             'concat_1_data': {}
                             })

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1_data', 'conv_1'),
                                 ('conv_1', 'conv_1_data'),
                                 ('const_conv_1_w', 'conv_1_w'),
                                 ('const_conv_1_b', 'conv_1_b'),
                                 ('conv_1_w', 'conv_1'),
                                 ('conv_1_b', 'conv_1'),
                                 ('conv_1_data', 'concat_1'),
                                 ('const_mul_1_w', 'mul_1_w'),
                                 ('mul_1_w', 'mul_1'),
                                 ('conv_1_data', 'mul_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('mul_1', 'mul_1_data'),
                                 ('mul_1_data', 'concat_1'),
                                 ('concat_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'const_conv_1_w': {'shape': np.array([1, 1, 3, 3]), 'value': np.zeros((1, 1, 3, 3))},
                                 'conv_1_w': {'shape': np.array([1, 1, 3, 3]), 'value': np.zeros((1, 1, 3, 3)),
                                              'output_channel_dim': 3, 'input_channel_dim': 2,
                                              'dims_number': 4},
                                 'const_conv_1_b': {'shape': np.array([3]), 'value': np.ones(3)},
                                 'conv_1_b': {'shape': np.array([3]), 'value': np.ones(3)},
                                 'conv_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'const_mul_1_w': {'shape': np.array([1]), 'value': np.array([6])},
                                 'mul_1_w': {'shape': np.array([1]), 'value': np.array([6])},
                                 'concat_1_data': {}
                                 })

        fuse_linear_ops(graph)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data')
        # TODO: refactor this test
        # self.assertTrue(flag, resp)
        

    # Op->Mul(array)-+->Conv(w+b)------+->Concat
    #                |                 |         =>  Same('can_be_fused': False)
    #                +-->Conv(w+b)-----+
    def test_fuse_lin_ops_2(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'conv_1'),
                             ('const_conv_1_w', 'conv_1_w'),
                             ('const_conv_1_b', 'conv_1_b'),
                             ('conv_1_w', 'conv_1'),
                             ('conv_1_b', 'conv_1'),
                             ('conv_1', 'conv_1_data'),
                             ('mul_1_data', 'conv_2'),
                             ('const_conv_2_w', 'conv_2_w'),
                             ('const_conv_2_b', 'conv_2_b'),
                             ('conv_2_w', 'conv_2'),
                             ('conv_2_b', 'conv_2'),
                             ('conv_2', 'conv_2_data'),
                             ('conv_1_data', 'concat_1'),
                             ('conv_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('concat_1_data', 'op_output')

                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'const_mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'const_conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96))},
                             'conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96)),
                                          'output_channel_dim': 3, 'input_channel_dim': 2,
                                          'dims_number': 4},
                             'const_conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                             'conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                             'conv_1_data': {'shape': np.array([1, 55, 55, 96])},
                             'conv_2': {'can_be_fused': False},
                             'const_conv_2_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96))},
                             'conv_2_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96)),
                                          'output_channel_dim': 3, 'input_channel_dim': 2,
                                          'dims_number': 4},
                             'const_conv_2_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                             'conv_2_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                             'conv_2_data': {'shape': np.array([1, 55, 55, 96])},
                             'concat_1_data': {}
                             })

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('const_mul_1_w', 'mul_1_w'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'mul_1_data'),
                                 ('mul_1_data', 'conv_1'),
                                 ('const_conv_1_w', 'conv_1_w'),
                                 ('const_conv_1_b', 'conv_1_b'),
                                 ('conv_1_w', 'conv_1'),
                                 ('conv_1_b', 'conv_1'),
                                 ('conv_1', 'conv_1_data'),
                                 ('mul_1_data', 'conv_2'),
                                 ('const_conv_2_w', 'conv_2_w'),
                                 ('const_conv_2_b', 'conv_2_b'),
                                 ('conv_2_w', 'conv_2'),
                                 ('conv_2_b', 'conv_2'),
                                 ('conv_2', 'conv_2_data'),
                                 ('conv_1_data', 'concat_1'),
                                 ('conv_2_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('concat_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'const_mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'const_conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96))},
                                 'conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96)),
                                              'output_channel_dim': 3, 'input_channel_dim': 2,
                                              'dims_number': 4},
                                 'const_conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_1_data': {'shape': np.array([1, 55, 55, 96])},
                                 'conv_2': {'can_be_fused': False},
                                 'const_conv_2_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96))},
                                 'conv_2_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96)),
                                              'output_channel_dim': 3, 'input_channel_dim': 2,
                                              'dims_number': 4},
                                 'const_conv_2_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_2_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_2_data': {'shape': np.array([1, 55, 55, 96])},
                                 'concat_1_data': {}
                                 })

        fuse_linear_ops(graph)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data')
        self.assertTrue(flag, resp)

    # Op->Mul(array)-+->Conv(w+b)------+->Concat
    #                |                 |         =>  Same('can_be_fused': False)
    #                +-->Conv(w+b)-----+
    def test_fuse_lin_ops_3(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'conv_1'),
                             ('const_conv_1_w', 'conv_1_w'),
                             ('const_conv_1_b', 'conv_1_b'),
                             ('conv_1_w', 'conv_1'),
                             ('conv_1_b', 'conv_1'),
                             ('conv_1', 'conv_1_data'),
                             ('mul_1_data', 'conv_2'),
                             ('const_conv_2_w', 'conv_2_w'),
                             ('const_conv_2_b', 'conv_2_b'),
                             ('conv_2_w', 'conv_2'),
                             ('conv_2_b', 'conv_2'),
                             ('conv_2', 'conv_2_data'),
                             ('conv_1_data', 'concat_1'),
                             ('conv_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('concat_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1': {'can_be_fused': False},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'const_mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'const_conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96))},
                             'conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96)),
                                          'output_channel_dim': 3, 'input_channel_dim': 2,
                                          'dims_number': 4},
                             'const_conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                             'conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                             'conv_1_data': {'shape': np.array([1, 55, 55, 96])},
                             'const_conv_2_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96))},
                             'conv_2_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96)),
                                          'output_channel_dim': 3, 'input_channel_dim': 2,
                                          'dims_number': 4},
                             'const_conv_2_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                             'conv_2_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                             'conv_2_data': {'shape': np.array([1, 55, 55, 96])},
                             'concat_1_data': {}
                             })

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('const_mul_1_w', 'mul_1_w'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'mul_1_data'),
                                 ('mul_1_data', 'conv_1'),
                                 ('const_conv_1_w', 'conv_1_w'),
                                 ('const_conv_1_b', 'conv_1_b'),
                                 ('conv_1_w', 'conv_1'),
                                 ('conv_1_b', 'conv_1'),
                                 ('conv_1', 'conv_1_data'),
                                 ('mul_1_data', 'conv_2'),
                                 ('const_conv_2_w', 'conv_2_w'),
                                 ('const_conv_2_b', 'conv_2_b'),
                                 ('conv_2_w', 'conv_2'),
                                 ('conv_2_b', 'conv_2'),
                                 ('conv_2', 'conv_2_data'),
                                 ('conv_1_data', 'concat_1'),
                                 ('conv_2_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('concat_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1': {'can_be_fused': False},
                                 'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'const_mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'const_conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96))},
                                 'conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96)),
                                              'output_channel_dim': 3, 'input_channel_dim': 2,
                                              'dims_number': 4},
                                 'const_conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_1_data': {'shape': np.array([1, 55, 55, 96])},
                                 'const_conv_2_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96))},
                                 'conv_2_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96)),
                                              'output_channel_dim': 3, 'input_channel_dim': 2,
                                              'dims_number': 4},
                                 'const_conv_2_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_2_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_2_data': {'shape': np.array([1, 55, 55, 96])},
                                 'concat_1_data': {}
                                 })

        fuse_linear_ops(graph)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data')
        self.assertTrue(flag, resp)
