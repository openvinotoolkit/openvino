# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from mo.front.common.partial_infer.eltwise import eltwise_infer
from mo.graph.graph import Node
from mo.middle.passes.fusing.fuse_linear_ops import _fuse_mul, fuse_linear_ops
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

nodes_attributes = {
    'placeholder_1': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # ScaleShift layer
    'scaleshift_1': {'type': 'ScaleShift', 'kind': 'op', 'op': 'ScaleShift'},
    'scaleshift_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'scaleshift_1_b': {'value': None, 'shape': None, 'kind': 'data'},
    'const_scaleshift_1_w': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
    'const_scaleshift_1_b': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
    'scaleshift_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Mul and Add operations
    'mul_1': {'type': 'Mul', 'kind': 'op', 'op': 'Mul', 'can_be_fused': True,
              'infer': lambda node: eltwise_infer(node, lambda a, b: a * b)},
    'mul_1_w': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'const_mul_1_w': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
    'mul_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'add_1': {'type': 'Add', 'kind': 'op', 'op': 'Add', 'can_be_fused': True},
    'add_1_w': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'const_add_1_w': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
    'add_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # Mul2 and Add2 operations
    'mul_2': {'type': 'Mul', 'kind': 'op', 'op': 'Mul', 'can_be_fused': True},
    'mul_2_w': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'const_mul_2_w': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
    'mul_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'add_2': {'type': 'Add', 'kind': 'op', 'op': 'Add', 'can_be_fused': True},
    'add_2_w': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'const_add_2_w': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
    'add_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # Concat1 operation
    'concat_1': {'type': 'Concat', 'kind': 'op', 'op': 'Concat'},
    'concat_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Convolutions
    'conv_1': {'type': 'Convolution', 'kind': 'op', 'op': 'Conv2D', 'layout': 'NHWC'},
    'conv_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_1_b': {'value': None, 'shape': None, 'kind': 'data'},
    'const_conv_1_w': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
    'const_conv_1_b': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
    'conv_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_2': {'type': 'Convolution', 'kind': 'op', 'op': 'Conv2D', 'layout': 'NHWC'},
    'conv_2_w': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_2_b': {'value': None, 'shape': None, 'kind': 'data'},
    'const_conv_2_w': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
    'const_conv_2_b': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
    'conv_2_data': {'value': None, 'shape': None, 'kind': 'data'},
    'deconv': {'type': 'Deconvolution', 'kind': 'op', 'op': 'Deconv2D', 'layout': 'NHWC'},
    'deconv_w': {'value': None, 'shape': None, 'kind': 'data'},
    'deconv_b': {'value': None, 'shape': None, 'kind': 'data'},
    'const_deconv_w': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
    'const_deconv_b': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
    'deconv_data': {'value': None, 'shape': None, 'kind': 'data'},
    # MatMul
    'fc_1': {'type': 'MatMul', 'kind': 'op', 'layout': 'NHWC', 'op': 'MatMul'},
    'fc_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'fc_1_b': {'value': None, 'shape': None, 'kind': 'data'},
    'const_fc_1_w': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
    'const_fc_1_b': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
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
                                 'const_conv_1_w': {'shape': np.array([11, 11, 3, 96]),
                                                    'value': np.ones((11, 11, 3, 96))},
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

    # Deconv(w)->Mul(array)
    def test_fuse_mul_to_deconv_1(self):
        # Placeholder->Deonv->Mul
        in_shape = np.array([1, 20, 10, 10])
        w_shape = np.array([20, 2, 3, 3])
        out_shape = np.array([1, 10, 21, 21])
        mul_const = np.array(range(10))

        edges = [('placeholder_1', 'placeholder_1_data'),
                 ('placeholder_1_data', 'deconv'),
                 ('const_deconv_w', 'deconv_w'),
                 ('deconv_w', 'deconv'),
                 ('deconv', 'deconv_data'),
                 ('deconv_data', 'mul_1'),
                 ('const_mul_1_w', 'mul_1_w'),
                 ('mul_1_w', 'mul_1'),
                 ('mul_1', 'mul_1_data'),
                 ('mul_1_data', 'op_output')
                 ]
        attr_updates = {'placeholder_1_data': {'shape': in_shape},
                        'const_conv_1_w': {'shape': w_shape, 'value': np.ones(w_shape)},
                        'deconv': {'group': 5},
                        'deconv_w': {'shape': w_shape, 'value': np.ones(w_shape),
                                     'output_channel_dim': 1, 'input_channel_dim': 0,
                                     'dims_number': 4},
                        'deconv_data': {'shape': out_shape},
                        'mul_1_data': {'shape': mul_const.shape},
                        'const_mul_1_w': {'shape': mul_const.shape, 'value': mul_const},
                        'mul_1_w': {'shape': mul_const.shape, 'value': mul_const},
                        }
        graph = build_graph(nodes_attributes, edges, attr_updates)
        # same graph, nothing fused
        graph_ref = build_graph(nodes_attributes, edges, attr_updates)

        _fuse_mul(graph, Node(graph, 'mul_1'), [Node(graph, 'deconv')], backward=True)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1', 'placeholder_1')
        self.assertTrue(flag, resp)

    def test_fuse_mul_data_nodes_names(self):
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
                             'const_conv_1_w': {'shape': np.array([11, 11, 3, 96]),
                                                'value': np.ones((11, 11, 3, 96))},
                             'conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96)),
                                          'output_channel_dim': 3, 'input_channel_dim': 2,
                                          'dims_number': 4},
                             'const_conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                             'conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                             'conv_1_data': {}
                             })

        _fuse_mul(graph, Node(graph, 'mul_1'), [Node(graph, 'conv_1')], backward=False)

        conv_node = Node(graph, 'conv_1')
        conv_in_data_name = conv_node.in_node(1)['name']
        const_node = Node(graph, 'const_conv_1_w')
        const_out_data_name = const_node.out_node(0)['name']
        mul_node = Node(graph, 'mul_1')
        conv_in_data = conv_node.in_node(1)

        # Check that transformation doesn't produce identical data node names,
        # as this may lead to appearing of Const ops with identical names.
        self.assertFalse(conv_in_data_name == const_out_data_name)

        # Attributes that are required for fusing are kept on data nodes.
        # These checks are needed to ensure that _fuse_mul doesn't remove any of these attributes.
        self.assertTrue(conv_in_data['output_channel_dim'] == 3)
        self.assertTrue(conv_in_data['input_channel_dim'] == 2)
        self.assertTrue(conv_in_data['dims_number'] == 4)
        self.assertTrue(mul_node['can_be_fused'] is True)


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
                                 'const_conv_1_w': {'shape': np.array([11, 11, 3, 96]),
                                                    'value': np.ones((11, 11, 3, 96))},
                                 'conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96)),
                                              'output_channel_dim': 3, 'input_channel_dim': 2,
                                              'dims_number': 4},
                                 'const_conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_1_data': {'shape': np.array([1, 55, 55, 96])},
                                 'conv_2': {'can_be_fused': False},
                                 'const_conv_2_w': {'shape': np.array([11, 11, 3, 96]),
                                                    'value': np.ones((11, 11, 3, 96))},
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
                                 'const_conv_1_w': {'shape': np.array([11, 11, 3, 96]),
                                                    'value': np.ones((11, 11, 3, 96))},
                                 'conv_1_w': {'shape': np.array([11, 11, 3, 96]), 'value': np.ones((11, 11, 3, 96)),
                                              'output_channel_dim': 3, 'input_channel_dim': 2,
                                              'dims_number': 4},
                                 'const_conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_1_b': {'shape': np.array([96]), 'value': np.zeros(96)},
                                 'conv_1_data': {'shape': np.array([1, 55, 55, 96])},
                                 'const_conv_2_w': {'shape': np.array([11, 11, 3, 96]),
                                                    'value': np.ones((11, 11, 3, 96))},
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
