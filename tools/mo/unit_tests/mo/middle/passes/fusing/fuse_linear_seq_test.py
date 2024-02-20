# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.middle.passes.fusing.fuse_linear_seq import fuse_mul_add_sequence
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

nodes_attributes = {
    'placeholder_1': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # ScaleShift layer
    'scaleshift_1': {'type': 'ScaleShift', 'kind': 'op', 'op': 'ScaleShift'},
    'const_scaleshift_1_w': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
    'scaleshift_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'const_scaleshift_1_b': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
    'scaleshift_1_b': {'value': None, 'shape': None, 'kind': 'data'},
    'scaleshift_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Mul and Add operations
    'mul_1': {'type': 'Mul', 'kind': 'op', 'op': 'Mul', 'can_be_fused': True},
    'const_mul_1_w': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
    'mul_1_w': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'mul_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'add_1': {'type': 'Add', 'kind': 'op', 'op': 'Add', 'can_be_fused': True},
    'const_add_1_w': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
    'add_1_w': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'add_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # Mul2 and Add2 operations
    'mul_2': {'type': 'Mul', 'kind': 'op', 'op': 'Mul', 'can_be_fused': True},
    'const_mul_2_w': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
    'mul_2_w': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'mul_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'add_2': {'type': 'Add', 'kind': 'op', 'op': 'Add', 'can_be_fused': True},
    'const_add_2_w': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
    'add_2_w': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'add_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # Mul3 and Add3 operations
    'mul_3': {'type': 'Mul', 'kind': 'op', 'op': 'Mul', 'can_be_fused': True},
    'const_mul_3_w': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
    'mul_3_w': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'mul_3_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'add_3': {'type': 'Add', 'kind': 'op', 'op': 'Add', 'can_be_fused': True},
    'const_add_3_w': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
    'add_3_w': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'add_3_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # Mul4 and Add4 operations
    'mul_4': {'type': 'Mul', 'kind': 'op', 'op': 'Mul', 'can_be_fused': True},
    'const_mul_4_w': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
    'mul_4_w': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'mul_4_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'add_4': {'type': 'Add', 'kind': 'op', 'op': 'Add', 'can_be_fused': True},
    'const_add_4_w': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
    'add_4_w': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'add_4_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # Concat1 operation
    'concat_1': {'type': 'Concat', 'kind': 'op', 'op': 'Concat'},
    'concat_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Convolutions
    'conv_1': {'type': 'Convolution', 'kind': 'op', 'op': 'Conv2D', 'layout': 'NHWC'},
    'const_conv_1_w': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
    'conv_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'const_conv_1_b': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
    'conv_1_b': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_2': {'type': 'Convolution', 'kind': 'op', 'op': 'Conv2D', 'layout': 'NHWC'},
    'const_conv_2_w': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
    'conv_2_w': {'value': None, 'shape': None, 'kind': 'data'},
    'const_conv_2_b': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
    'conv_2_b': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_2_data': {'value': None, 'shape': None, 'kind': 'data'},
    # FullyConnected
    'fc_1': {'type': 'MatMul', 'kind': 'op', 'op': 'FullyConnected', 'layout': 'NHWC'},
    'const_fc_1_w': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
    'fc_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'const_fc_1_b': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'op': 'Const'},
    'fc_1_b': {'value': None, 'shape': None, 'kind': 'data'},
    'fc_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Placeholders
    'placeholder_2': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_3': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_3_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'op_output': {'kind': 'op', 'op': 'Result'}
}


# Unit tests for fuse_mul_add_sequence
class LinSeqFusingTests(unittest.TestCase):
    # Placeholder-+->Mul->Add->Mul-+->Concat
    #             |                |
    #             +----------------+
    def test_fuse_lin_seq_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('const_add_1_w', 'add_1_w'),
                             ('add_1_w', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'mul_2'),
                             ('const_mul_2_w', 'mul_2_w'),
                             ('mul_2_w', 'mul_2'),
                             ('mul_2', 'mul_2_data'),
                             ('mul_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('placeholder_1_data', 'concat_1'),
                             ('concat_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_2_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_w': {'shape': np.array([1]), 'value': 6},
                             'add_1_w': {'shape': np.array([1]), 'value': 6},
                             'mul_2_w': {'shape': np.array([1]), 'value': 6},
                             },
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('const_mul_1_w', 'mul_1_w'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'mul_1_data'),
                                 ('mul_1_data', 'add_1'),
                                 ('const_add_1_w', 'add_1_w'),
                                 ('add_1_w', 'add_1'),
                                 ('add_1', 'add_1_data'),
                                 ('add_1_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('placeholder_1_data', 'concat_1'),
                                 ('concat_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_w': {'shape': np.array([1]), 'value': np.array([36])},
                                 'add_1_w': {'shape': np.array([1]), 'value': np.array([36])},
                                 'mul_1': {'can_be_fused': True},
                                 'add_1': {'can_be_fused': True},
                                 },
                                nodes_with_edges_only=True)

        graph.graph['layout'] = 'NHWC'
        fuse_mul_add_sequence(graph)
        graph.clean_up()
        self.assertTrue(len(graph.node) == len(graph_ref.node),
                       "Graphs has different number of nodes: {} and {}".format(len(graph.node), len(graph_ref.node)))

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data')
        self.assertTrue(flag, resp)

    #             +----------------+
    #             |                |
    # Placeholder-+->Mul->Add->Mul-+---------------+->Concat
    #                           |                  |
    #                           +-->Placeholder----+
    def test_fuse_lin_seq_2(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('const_add_1_w', 'add_1_w'),
                             ('add_1_w', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'mul_2'),
                             ('const_mul_2_w', 'mul_2_w'),
                             ('mul_2_w', 'mul_2'),
                             ('mul_2', 'mul_2_data'),
                             ('mul_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('placeholder_1_data', 'concat_1'),
                             ('mul_2_data', 'placeholder_2'),
                             ('placeholder_2', 'placeholder_2_data'),
                             ('placeholder_2_data', 'concat_1'),
                             ('concat_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'placeholder_2_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_2_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_w': {'shape': np.array([1]), 'value': 6},
                             'add_1_w': {'shape': np.array([1]), 'value': 6},
                             'mul_2_w': {'shape': np.array([1]), 'value': 6},
                             },
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('const_mul_1_w', 'mul_1_w'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'mul_1_data'),
                                 ('mul_1_data', 'add_1'),
                                 ('const_add_1_w', 'add_1_w'),
                                 ('add_1_w', 'add_1'),
                                 ('add_1', 'add_1_data'),
                                 ('add_1_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('placeholder_1_data', 'concat_1'),
                                 ('add_1_data', 'placeholder_2'),
                                 ('placeholder_2', 'placeholder_2_data'),
                                 ('placeholder_2_data', 'concat_1'),
                                 ('concat_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'placeholder_2_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_w': {'shape': np.array([1]), 'value': np.array([36])},
                                 'add_1_w': {'shape': np.array([1]), 'value': np.array([36])},
                                 'mul_1': {'can_be_fused': True},
                                 'add_1': {'can_be_fused': True},
                                 },
                                nodes_with_edges_only=True)
        graph.graph['layout'] = 'NHWC'
        fuse_mul_add_sequence(graph)
        graph.clean_up()
        self.assertTrue(len(graph.node) == len(graph_ref.node),
                        "Graphs has different number of nodes: {} and {}".format(len(graph.node), len(graph_ref.node)))

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data')
        self.assertTrue(flag, resp)

    #                      +----->Placeholder
    #                      |        |          =>  The same graph
    # Placeholder--->Mul->Add->Mul--+->Concat
    def test_fuse_lin_seq_3(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('const_add_1_w', 'add_1_w'),
                             ('add_1_w', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'mul_2'),
                             ('const_mul_2_w', 'mul_2_w'),
                             ('mul_2_w', 'mul_2'),
                             ('mul_2', 'mul_2_data'),
                             ('mul_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('add_1_data', 'placeholder_2'),
                             ('placeholder_2', 'placeholder_2_data'),
                             ('placeholder_2_data', 'concat_1'),
                             ('concat_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'placeholder_2_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_2_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_w': {'shape': np.array([]), 'value': np.array(6)},
                             'add_1_w': {'shape': np.array([]), 'value': np.array(6)},
                             'mul_2_w': {'shape': np.array([]), 'value': np.array(6)},
                             },
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('const_mul_1_w', 'mul_1_w'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'mul_1_data'),
                                 ('mul_1_data', 'add_1'),
                                 ('const_add_1_w', 'add_1_w'),
                                 ('add_1_w', 'add_1'),
                                 ('add_1', 'add_1_data'),
                                 ('add_1_data', 'mul_2'),
                                 ('const_mul_2_w', 'mul_2_w'),
                                 ('mul_2_w', 'mul_2'),
                                 ('mul_2', 'mul_2_data'),
                                 ('mul_2_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('add_1_data', 'placeholder_2'),
                                 ('placeholder_2', 'placeholder_2_data'),
                                 ('placeholder_2_data', 'concat_1'),
                                 ('concat_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'placeholder_2_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_2_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_w': {'shape': np.array([]), 'value': np.array(6)},
                                 'add_1_w': {'shape': np.array([]), 'value': np.array(6)},
                                 'mul_2_w': {'shape': np.array([]), 'value': np.array(6)},
                                 },
                                nodes_with_edges_only=True)

        fuse_mul_add_sequence(graph)
        graph.clean_up()
        self.assertTrue(len(graph.node) == len(graph_ref.node),
                        "Graphs has different number of nodes: {} and {}".format(len(graph.node), len(graph_ref.node)))

        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1')
        self.assertTrue(flag, resp)

    #                 +-------->Placeholder                          +-------->Placeholder
    #                 |            |           =>                    |            |
    # Placeholder--->Mul->Add->Mul-+->Concat         Placeholder-+->Mul->Mul->Add-+->Concat
    def test_fuse_lin_seq_4(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('const_add_1_w', 'add_1_w'),
                             ('add_1_w', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'mul_2'),
                             ('const_mul_2_w', 'mul_2_w'),
                             ('mul_2_w', 'mul_2'),
                             ('mul_2', 'mul_2_data'),
                             ('mul_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('mul_1_data', 'placeholder_2'),
                             ('placeholder_2', 'placeholder_2_data'),
                             ('placeholder_2_data', 'concat_1'),
                             ('concat_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'placeholder_2_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_2_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_w': {'shape': np.array([]), 'value': np.array(6)},
                             'add_1_w': {'shape': np.array([]), 'value': np.array(6)},
                             'mul_2_w': {'shape': np.array([]), 'value': np.array(6)},
                             },
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('const_mul_1_w', 'mul_1_w'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'mul_1_data'),
                                 ('mul_1_data', 'mul_2'),
                                 ('const_mul_2_w', 'mul_2_w'),
                                 ('mul_2_w', 'mul_2'),
                                 ('mul_2', 'mul_2_data'),
                                 ('mul_2_data', 'add_1'),
                                 ('const_add_1_w', 'add_1_w'),
                                 ('add_1_w', 'add_1'),
                                 ('add_1', 'add_1_data'),
                                 ('add_1_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('mul_1_data', 'placeholder_2'),
                                 ('placeholder_2', 'placeholder_2_data'),
                                 ('placeholder_2_data', 'concat_1'),
                                 ('concat_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'placeholder_2_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_2_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_w': {'shape': np.array([]), 'value': np.array([6])},
                                 'add_1_w': {'shape': np.array([1]), 'value': np.array([36])},
                                 'mul_2_w': {'shape': np.array([1]), 'value': np.array([6])},
                                 },
                                nodes_with_edges_only=True)

        graph.graph['layout'] = 'NHWC'
        fuse_mul_add_sequence(graph)
        graph.clean_up()
        self.assertTrue(len(graph.node) == len(graph_ref.node),
                        "Graphs has different number of nodes: {} and {}".format(len(graph.node), len(graph_ref.node)))

        (flag, resp) = compare_graphs(graph, graph_ref, 'op_output')
        self.assertTrue(flag, resp)

    #                 +-------->Placeholder                          +->Placeholder
    #                 |            |           =>                    |            |
    # Placeholder--->Mul->Add->Mul-+->Concat         Placeholder--->Mul-----------+->Concat
    def test_fuse_lin_seq_5(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('const_add_1_w', 'add_1_w'),
                             ('add_1_w', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'mul_2'),
                             ('const_mul_2_w', 'mul_2_w'),
                             ('mul_2_w', 'mul_2'),
                             ('mul_2', 'mul_2_data'),
                             ('mul_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('mul_1_data', 'placeholder_2'),
                             ('placeholder_2', 'placeholder_2_data'),
                             ('placeholder_2_data', 'concat_1'),
                             ('concat_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'placeholder_2_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_2_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_w': {'shape': np.array([]), 'value': np.array(6)},
                             'add_1_w': {'shape': np.array([]), 'value': np.array(0)},
                             'mul_2_w': {'shape': np.array([]), 'value': np.array(1)},
                             },
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('const_mul_1_w', 'mul_1_w'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'mul_1_data'),
                                 ('mul_1_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('mul_1_data', 'placeholder_2'),
                                 ('placeholder_2', 'placeholder_2_data'),
                                 ('placeholder_2_data', 'concat_1'),
                                 ('concat_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'placeholder_2_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_w': {'shape': np.array([]), 'value': np.array([6])},
                                 },
                                nodes_with_edges_only=True)

        graph.graph['layout'] = 'NHWC'
        fuse_mul_add_sequence(graph)
        graph.clean_up()

        self.assertTrue(len(graph.node) == len(graph_ref.node),
                        "Graphs has different number of nodes: {} and {}".format(len(graph.node), len(graph_ref.node)))

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data')
        self.assertTrue(flag, resp)

    #                 +-------->Placeholder                          +->Placeholder
    #                 |            |           =>                    |            |
    # Placeholder--->Mul->Add->Mul-+->Concat         Placeholder--->Mul-->Add-----+->Concat
    def test_fuse_lin_seq_6(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('const_add_1_w', 'add_1_w'),
                             ('add_1_w', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'mul_2'),
                             ('const_mul_2_w', 'mul_2_w'),
                             ('mul_2_w', 'mul_2'),
                             ('mul_2', 'mul_2_data'),
                             ('mul_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('mul_1_data', 'placeholder_2'),
                             ('placeholder_2', 'placeholder_2_data'),
                             ('placeholder_2_data', 'concat_1'),
                             ('concat_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'placeholder_2_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_2_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_w': {'shape': np.array([]), 'value': np.array(6)},
                             'add_1_w': {'shape': np.array([]), 'value': np.array(6)},
                             'mul_2_w': {'shape': np.array([]), 'value': np.array(1)},
                             },
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('const_mul_1_w', 'mul_1_w'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'mul_1_data'),
                                 ('mul_1_data', 'add_1'),
                                 ('const_add_1_w', 'add_1_w'),
                                 ('add_1_w', 'add_1'),
                                 ('add_1', 'add_1_data'),
                                 ('add_1_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('mul_1_data', 'placeholder_2'),
                                 ('placeholder_2', 'placeholder_2_data'),
                                 ('placeholder_2_data', 'concat_1'),
                                 ('concat_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'placeholder_2_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_w': {'shape': np.array([]), 'value': np.array(6)},
                                 'add_1_w': {'shape': np.array([1]), 'value': np.array([6])},
                                 },
                                nodes_with_edges_only=True)

        graph.graph['layout'] = 'NHWC'
        fuse_mul_add_sequence(graph)
        graph.clean_up()

        self.assertTrue(len(graph.node) == len(graph_ref.node),
                        "Graphs has different number of nodes: {} and {}".format(len(graph.node), len(graph_ref.node)))

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data')
        self.assertTrue(flag, resp)

    #                 +-------->Placeholder                          +->Placeholder
    #                 |            |           =>                    |            |
    # Placeholder--->Mul->Add->Mul-+->Concat         Placeholder--->Mul-->Mul-----+->Concat
    def test_fuse_lin_seq_7(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('const_add_1_w', 'add_1_w'),
                             ('add_1_w', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'mul_2'),
                             ('const_mul_2_w', 'mul_2_w'),
                             ('mul_2_w', 'mul_2'),
                             ('mul_2', 'mul_2_data'),
                             ('mul_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('mul_1_data', 'placeholder_2'),
                             ('placeholder_2', 'placeholder_2_data'),
                             ('placeholder_2_data', 'concat_1'),
                             ('concat_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'placeholder_2_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_2_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_w': {'shape': np.array([]), 'value': np.array(6)},
                             'add_1_w': {'shape': np.array([]), 'value': np.array(0)},
                             'mul_2_w': {'shape': np.array([]), 'value': np.array(6)},
                             },
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('const_mul_1_w', 'mul_1_w'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'mul_1_data'),
                                 ('mul_1_data', 'mul_2'),
                                 ('const_mul_2_w', 'mul_2_w'),
                                 ('mul_2_w', 'mul_2'),
                                 ('mul_2', 'mul_2_data'),
                                 ('mul_2_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('mul_1_data', 'placeholder_2'),
                                 ('placeholder_2', 'placeholder_2_data'),
                                 ('placeholder_2_data', 'concat_1'),
                                 ('concat_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'placeholder_2_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_w': {'shape': np.array([]), 'value': np.array(6)},
                                 'mul_2_w': {'shape': np.array([1]), 'value': np.array([6])},
                                 },
                                nodes_with_edges_only=True)

        graph.graph['layout'] = 'NHWC'
        fuse_mul_add_sequence(graph)
        graph.clean_up()
        self.assertTrue(len(graph.node) == len(graph_ref.node),
                        "Graphs has different number of nodes: {} and {}".format(len(graph.node), len(graph_ref.node)))

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data')
        self.assertTrue(flag, resp)

    # Placeholder--->Mul->Add->Mul-+->Concat         Placeholder->Concat
    def test_fuse_lin_seq_8(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('const_add_1_w', 'add_1_w'),
                             ('add_1_w', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'mul_2'),
                             ('const_mul_2_w', 'mul_2_w'),
                             ('mul_2_w', 'mul_2'),
                             ('mul_2', 'mul_2_data'),
                             ('mul_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('concat_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_2_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_w': {'shape': np.array([]), 'value': np.array(1)},
                             'add_1_w': {'shape': np.array([]), 'value': np.array(0)},
                             'mul_2_w': {'shape': np.array([]), 'value': np.array(1)},
                             },
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('concat_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])}},
                                nodes_with_edges_only=True)

        graph.graph['layout'] = 'NHWC'
        fuse_mul_add_sequence(graph)
        graph.clean_up()
        self.assertTrue(len(graph.node) == len(graph_ref.node),
                        "Graphs has different number of nodes: {} and {}".format(len(graph.node), len(graph_ref.node)))

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data')
        self.assertTrue(flag, resp)

    # Placeholder--->Mul->Add->Mul-+->Concat         Placeholder->Mul->Add->Concat
    def test_fuse_lin_seq_9(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('const_add_1_w', 'add_1_w'),
                             ('add_1_w', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'mul_2'),
                             ('const_mul_2_w', 'mul_2_w'),
                             ('mul_2_w', 'mul_2'),
                             ('mul_2', 'mul_2_data'),
                             ('mul_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('concat_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_2_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_w': {'shape': np.array([1]), 'value': np.array(6)},
                             'add_1_w': {'shape': np.array([1]), 'value': np.array(6)},
                             'mul_2_w': {'shape': np.array([1]), 'value': np.array(6)},
                             },
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('const_mul_1_w', 'mul_1_w'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'mul_1_data'),
                                 ('mul_1_data', 'add_1'),
                                 ('const_add_1_w', 'add_1_w'),
                                 ('add_1_w', 'add_1'),
                                 ('add_1', 'add_1_data'),
                                 ('add_1_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('concat_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_w': {'shape': np.array([1]), 'value': np.array([36])},
                                 'add_1_w': {'shape': np.array([1]), 'value': np.array([36])},
                                 },
                                nodes_with_edges_only=True)

        graph.graph['layout'] = 'NHWC'
        fuse_mul_add_sequence(graph)
        graph.clean_up()
        self.assertTrue(len(graph.node) == len(graph_ref.node),
                        "Graphs has different number of nodes: {} and {}".format(len(graph.node), len(graph_ref.node)))

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data')
        self.assertTrue(flag, resp)

    # Placeholder--->Mul->Add->Mul-+->Concat         Placeholder->Mul->Add->Concat
    def test_fuse_lin_seq_10(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('const_add_1_w', 'add_1_w'),
                             ('add_1_w', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'mul_2'),
                             ('const_mul_2_w', 'mul_2_w'),
                             ('mul_2_w', 'mul_2'),
                             ('mul_2', 'mul_2_data'),
                             ('mul_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('concat_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_2_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_w': {'shape': np.array([]), 'value': np.array(6)},
                             'add_1_w': {'shape': np.array([3]), 'value': np.array([6, 6, 6])},
                             'mul_2_w': {'shape': np.array([]), 'value': np.array(6)},
                             },
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('const_mul_1_w', 'mul_1_w'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'mul_1_data'),
                                 ('mul_1_data', 'add_1'),
                                 ('const_add_1_w', 'add_1_w'),
                                 ('add_1_w', 'add_1'),
                                 ('add_1', 'add_1_data'),
                                 ('add_1_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('concat_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_w': {'shape': np.array([3]), 'value': np.array([36, 36, 36])},
                                 'add_1_w': {'shape': np.array([3]), 'value': np.array([36, 36, 36])},
                                 },
                                nodes_with_edges_only=True)

        graph.graph['layout'] = 'NHWC'
        fuse_mul_add_sequence(graph)
        graph.clean_up()
        self.assertTrue(len(graph.node) == len(graph_ref.node),
                        "Graphs has different number of nodes: {} and {}".format(len(graph.node), len(graph_ref.node)))

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data')
        self.assertTrue(flag, resp)

    # Placeholder-+->Mul->Add->Mul-+->Concat
    #             |                |            With 'can_be_fused' = False
    #             +----------------+
    def test_fuse_lin_seq_11(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('const_add_1_w', 'add_1_w'),
                             ('add_1_w', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'mul_2'),
                             ('const_mul_2_w', 'mul_2_w'),
                             ('mul_2_w', 'mul_2'),
                             ('mul_2', 'mul_2_data'),
                             ('mul_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('placeholder_1_data', 'concat_1'),
                             ('concat_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_2_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_w': {'shape': np.array([]), 'value': np.array(6)},
                             'add_1_w': {'shape': np.array([]), 'value': np.array(6)},
                             'mul_2_w': {'shape': np.array([]), 'value': np.array(6)},
                             'mul_1': {'can_be_fused': False},
                             'add_1': {'can_be_fused': False},
                             },
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('const_mul_1_w', 'mul_1_w'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'mul_1_data'),
                                 ('mul_1_data', 'add_1'),
                                 ('const_add_1_w', 'add_1_w'),
                                 ('add_1_w', 'add_1'),
                                 ('add_1', 'add_1_data'),
                                 ('add_1_data', 'mul_2'),
                                 ('const_mul_2_w', 'mul_2_w'),
                                 ('mul_2_w', 'mul_2'),
                                 ('mul_2', 'mul_2_data'),
                                 ('mul_2_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('placeholder_1_data', 'concat_1'),
                                 ('concat_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_2_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_w': {'shape': np.array([]), 'value': np.array(6)},
                                 'add_1_w': {'shape': np.array([]), 'value': np.array(6)},
                                 'mul_2_w': {'shape': np.array([]), 'value': np.array(6)},
                                 'mul_1': {'can_be_fused': False},
                                 'add_1': {'can_be_fused': False},
                                 },
                                nodes_with_edges_only=True)

        graph.graph['layout'] = 'NHWC'
        fuse_mul_add_sequence(graph)
        graph.clean_up()
        self.assertTrue(len(graph.node) == len(graph_ref.node),
                        "Graphs has different number of nodes: {} and {}".format(len(graph.node), len(graph_ref.node)))

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data')
        self.assertTrue(flag, resp)

    # Placeholder-+->Mul->Add->Mul-+->Concat
    #             |                |            With 'can_be_fused' = False
    #             +----------------+
    def test_fuse_lin_seq_12(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('const_add_1_w', 'add_1_w'),
                             ('add_1_w', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'mul_2'),
                             ('const_mul_2_w', 'mul_2_w'),
                             ('mul_2_w', 'mul_2'),
                             ('mul_2', 'mul_2_data'),
                             ('mul_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('placeholder_1_data', 'concat_1'),
                             ('concat_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_2_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_w': {'shape': np.array([]), 'value': np.array(6)},
                             'add_1_w': {'shape': np.array([]), 'value': np.array(6)},
                             'mul_2_w': {'shape': np.array([]), 'value': np.array(6)},
                             'add_1': {'can_be_fused': False},
                             },
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('const_mul_1_w', 'mul_1_w'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'mul_1_data'),
                                 ('mul_1_data', 'add_1'),
                                 ('const_add_1_w', 'add_1_w'),
                                 ('add_1_w', 'add_1'),
                                 ('add_1', 'add_1_data'),
                                 ('add_1_data', 'mul_2'),
                                 ('const_mul_2_w', 'mul_2_w'),
                                 ('mul_2_w', 'mul_2'),
                                 ('mul_2', 'mul_2_data'),
                                 ('mul_2_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('placeholder_1_data', 'concat_1'),
                                 ('concat_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_2_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_w': {'shape': np.array([]), 'value': np.array(6)},
                                 'add_1_w': {'shape': np.array([]), 'value': np.array(6)},
                                 'mul_2_w': {'shape': np.array([]), 'value': np.array(6)},
                                 'add_1': {'can_be_fused': False},
                                 },
                                nodes_with_edges_only=True)

        graph.graph['layout'] = 'NHWC'
        fuse_mul_add_sequence(graph)
        graph.clean_up()
        self.assertTrue(len(graph.node) == len(graph_ref.node),
                        "Graphs has different number of nodes: {} and {}".format(len(graph.node), len(graph_ref.node)))

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data')
        self.assertTrue(flag, resp)

    # Placeholder-+->Mul->Add->Mul-+->Concat
    #             |                |
    #             +->Mul->Mul->----+  (This Mul ops has shared weights with upper Mul ops)
    def test_fuse_lin_seq_shared_weights_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('const_add_1_w', 'add_1_w'),
                             ('add_1_w', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'mul_2'),
                             ('const_mul_2_w', 'mul_2_w'),
                             ('mul_2_w', 'mul_2'),
                             ('mul_2', 'mul_2_data'),
                             ('mul_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('placeholder_1_data', 'mul_3'),
                             ('mul_3', 'mul_3_data'),
                             ('mul_1_w', 'mul_3'),
                             ('mul_3_data', 'mul_4'),
                             ('mul_2_w', 'mul_4'),
                             ('mul_4', 'mul_4_data'),
                             ('mul_4_data', 'concat_1'),
                             ('concat_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_2_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_3_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_4_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_w': {'shape': np.array([]), 'value': np.array(6)},
                             'add_1_w': {'shape': np.array([]), 'value': np.array(6)},
                             'mul_2_w': {'shape': np.array([]), 'value': np.array(6)},
                             },
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('const_mul_1_w', 'mul_1_w'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'mul_1_data'),
                                 ('mul_1_data', 'add_1'),
                                 ('const_add_1_w', 'add_1_w'),
                                 ('add_1_w', 'add_1'),
                                 ('add_1', 'add_1_data'),
                                 ('add_1_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('placeholder_1_data', 'mul_3'),
                                 ('mul_3', 'mul_3_data'),
                                 ('const_mul_3_w', 'mul_3_w'),
                                 ('mul_3_w', 'mul_3'),
                                 ('mul_3_data', 'concat_1'),
                                 ('concat_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_3_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_w': {'shape': np.array([1]), 'value': np.array([36])},
                                 'mul_3_w': {'shape': np.array([1]), 'value': np.array([36])},
                                 'add_1_w': {'shape': np.array([1]), 'value': np.array([36])},
                                 'mul_1': {'can_be_fused': True},
                                 'add_1': {'can_be_fused': True},
                                 },
                                nodes_with_edges_only=True)

        graph.graph['layout'] = 'NHWC'
        fuse_mul_add_sequence(graph)
        graph.clean_up()
        self.assertTrue(len(graph.node) == len(graph_ref.node),
                        "Graphs has different number of nodes: {} and {}".format(len(graph.node),
                                                                                 len(graph_ref.node)))

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data')
        self.assertTrue(flag, resp)
