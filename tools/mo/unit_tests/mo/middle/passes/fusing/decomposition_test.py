# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.middle.passes.fusing.decomposition import convert_scale_shift_to_mul_add, convert_batch_norm
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

nodes_attributes = {
    'placeholder_1': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_2': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # ScaleShift layer
    'scaleshift_1': {'type': 'ScaleShift', 'kind': 'op', 'op': 'ScaleShift', 'axis': 0},
    'const_scaleshift_1_w': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Const'},
    'scaleshift_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'const_scaleshift_1_b': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Const'},
    'scaleshift_1_b': {'value': None, 'shape': None, 'kind': 'data'},
    'scaleshift_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Mul and Add operations
    'mul_1': {'type': None, 'value': None, 'kind': 'op', 'op': 'Mul'},
    'const_mul_1_w': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Const'},
    'mul_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'mul_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'add_1': {'type': None, 'kind': 'op', 'op': 'Add'},
    'const_add_1_w': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Const'},
    'add_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'add_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Mul and Add operations
    'mul_2': {'type': None, 'kind': 'op', 'op': 'Mul'},
    'const_mul_2_w': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Const'},
    'mul_2_w': {'value': None, 'shape': None, 'kind': 'data'},
    'mul_2_data': {'value': None, 'shape': None, 'kind': 'data'},
    'add_2': {'type': None, 'kind': 'op', 'op': 'Add'},
    'const_add_2_w': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Const'},
    'add_2_w': {'value': None, 'shape': None, 'kind': 'data'},
    'add_2_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Reshape
    'placeholder_2/Reshape_': {'type': 'Reshape', 'kind': 'op', 'op': 'Reshape'},
    'placeholder_2/Reshape_data': {'value': None, 'shape': None, 'kind': 'data'},
    'placeholder_2/Reshape_const': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': None},
    'placeholder_2/Reshape_const_data': {'kind': 'data', 'value': None, 'shape': None},
    # BatchNorm operation
    'bn_op': {'type': None, 'kind': 'op', 'op': 'BatchNorm', 'can_be_fused': True},
    'const_bn_const': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Const'},
    'bn_const': {'value': None, 'shape': None, 'kind': 'data'},
    'const_bn_beta': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Const'},
    'bn_beta': {'value': None, 'shape': None, 'kind': 'data'},
    'const_bn_mean': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Const'},
    'bn_mean': {'value': None, 'shape': None, 'kind': 'data'},
    'const_bn_var': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Const'},
    'bn_var': {'value': None, 'shape': None, 'kind': 'data'},
    'bn_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Concat1 operation
    'concat': {'type': 'Concat', 'kind': 'op', 'op': 'Concat'},
    'concat_data': {'value': None, 'shape': None, 'kind': 'data'},
    'op_output': {'kind': 'op', 'op': 'Result'}
}


class ScaleShiftToMulAdd(unittest.TestCase):
    # ScaleShift -> Mul
    def test_scaleshift_to_mul_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'scaleshift_1'),
                             ('const_scaleshift_1_w', 'scaleshift_1_w'),
                             ('scaleshift_1_w', 'scaleshift_1'),
                             ('scaleshift_1', 'scaleshift_1_data'),
                             ('scaleshift_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'scaleshift_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'scaleshift_1_data': {}
                             })

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('const_mul_1_w', 'mul_1_w'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'scaleshift_1_data'),
                                 ('scaleshift_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'const_mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'mul_1': {'can_be_fused': True},
                                 'scaleshift_1_data': {}
                                 })

        graph.graph['layout'] = 'NHWC'
        convert_scale_shift_to_mul_add(graph)
        graph.clean_up()
        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1')
        self.assertTrue(flag, resp)

    # ScaleShift  2 inputs-> Mul
    def test_scaleshift2_to_mul(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_2', 'placeholder_2_data'),
                             ('placeholder_1_data', 'scaleshift_1'),
                             ('placeholder_2_data', 'scaleshift_1'),
                             ('scaleshift_1', 'scaleshift_1_data'),
                             ('scaleshift_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'placeholder_2_data': {'shape': np.array([1, 227])},
                             'scaleshift_1_data': {}
                             })

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_2', 'placeholder_2_data'),
                                 ('placeholder_2_data', 'placeholder_2/Reshape_'),
                                 ('placeholder_2/Reshape_const', 'placeholder_2/Reshape_const_data'),
                                 ('placeholder_2/Reshape_const_data', 'placeholder_2/Reshape_'),
                                 ('placeholder_2/Reshape_', 'placeholder_2/Reshape_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('placeholder_2/Reshape_data', 'mul_1'),
                                 ('mul_1', 'scaleshift_1_data'),
                                 ('scaleshift_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'placeholder_2_data': {'shape': np.array([1, 227])},
                                 'placeholder_2/Reshape_const': {'value': np.array([1, 227, 1, 1]), 'shape': [4]},
                                 'placeholder_2/Reshape_const_data': {'value': np.array([1, 227, 1, 1]), 'shape': [4]},
                                 'placeholder_2/Reshape_data': {'shape': np.array([1, 227, 1, 1])},
                                 'mul_1': {'can_be_fused': True},
                                 'scaleshift_1_data': {}
                                 })

        graph.graph['layout'] = 'NHWC'
        convert_scale_shift_to_mul_add(graph)
        graph.clean_up()
        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1')
        self.assertTrue(flag, resp)

    # ScaleShift  2 inputs-> Mul (axis = 1)
    def test_scaleshift2_axis1_to_mul(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_2', 'placeholder_2_data'),
                             ('placeholder_1_data', 'scaleshift_1'),
                             ('placeholder_2_data', 'scaleshift_1'),
                             ('scaleshift_1', 'scaleshift_1_data'),
                             ('scaleshift_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'placeholder_2_data': {'shape': np.array([227])},
                             'scaleshift_1': {'axis': 1},
                             'scaleshift_1_data': {}
                             })

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_2', 'placeholder_2_data'),
                                 ('placeholder_2_data', 'placeholder_2/Reshape_'),
                                 ('placeholder_2/Reshape_const', 'placeholder_2/Reshape_const_data'),
                                 ('placeholder_2/Reshape_const_data', 'placeholder_2/Reshape_'),
                                 ('placeholder_2/Reshape_', 'placeholder_2/Reshape_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('placeholder_2/Reshape_data', 'mul_1'),
                                 ('mul_1', 'scaleshift_1_data'),
                                 ('scaleshift_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'placeholder_2_data': {'shape': np.array([227])},
                                 'placeholder_2/Reshape_const': {'value': np.array([1, 227, 1, 1]), 'shape': [4]},
                                 'placeholder_2/Reshape_const_data': {'value': np.array([1, 227, 1, 1]), 'shape': [4]},
                                 'placeholder_2/Reshape_data': {'shape': np.array([1, 227, 1, 1])},
                                 'mul_1': {'can_be_fused': True},
                                 'scaleshift_1_data': {}
                                 })

        graph.graph['layout'] = 'NHWC'
        convert_scale_shift_to_mul_add(graph)
        graph.clean_up()
        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1')
        self.assertTrue(flag, resp)

    # ScaleShift -> Mul (Zero biases)
    def test_scaleshift_to_mul_2(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'scaleshift_1'),
                             ('const_scaleshift_1_w', 'scaleshift_1_w'),
                             ('const_scaleshift_1_b', 'scaleshift_1_b'),
                             ('scaleshift_1_w', 'scaleshift_1'),
                             ('scaleshift_1_b', 'scaleshift_1'),
                             ('scaleshift_1', 'scaleshift_1_data'),
                             ('scaleshift_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'scaleshift_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'scaleshift_1_b': {'shape': np.array([3]), 'value': np.array([0, 0, 0])},
                             'scaleshift_1_data': {}
                             })

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('const_mul_1_w', 'mul_1_w'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'scaleshift_1_data'),
                                 ('scaleshift_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'const_mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'mul_1': {'can_be_fused': True},
                                 'scaleshift_1_data': {}
                                 })

        graph.graph['layout'] = 'NHWC'
        convert_scale_shift_to_mul_add(graph)
        graph.clean_up()
        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1')
        self.assertTrue(flag, resp)

    # ScaleShift -> Mul->Add
    def test_scaleshift_to_mul_add(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'scaleshift_1'),
                             ('const_scaleshift_1_w', 'scaleshift_1_w'),
                             ('const_scaleshift_1_b', 'scaleshift_1_b'),
                             ('scaleshift_1_w', 'scaleshift_1'),
                             ('scaleshift_1_b', 'scaleshift_1'),
                             ('scaleshift_1', 'scaleshift_1_data'),
                             ('scaleshift_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'scaleshift_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'scaleshift_1_b': {'shape': np.array([3]), 'value': np.array([3, 2, 1])},
                             'scaleshift_1_data': {}
                             })

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('const_mul_1_w', 'mul_1_w'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'mul_1_data'),
                                 ('mul_1_data', 'add_1'),
                                 ('const_add_1_w', 'add_1_w'),
                                 ('add_1_w', 'add_1'),
                                 ('add_1', 'scaleshift_1_data'),
                                 ('scaleshift_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'const_mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'const_add_1_w': {'shape': np.array([3]), 'value': np.array([3, 2, 1])},
                                 'add_1_w': {'shape': np.array([3]), 'value': np.array([3, 2, 1])},
                                 'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'add_1': {'can_be_fused': True},
                                 'mul_1': {'can_be_fused': True},
                                 'scaleshift_1_data': {}
                                 })

        graph.graph['layout'] = 'NHWC'
        convert_scale_shift_to_mul_add(graph)
        graph.clean_up()
        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1')
        self.assertTrue(flag, resp)

    # ScaleShift -> None (Zero weights and biases)
    def test_scaleshift_to_nothing(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'scaleshift_1'),
                             ('const_scaleshift_1_w', 'scaleshift_1_w'),
                             ('const_scaleshift_1_b', 'scaleshift_1_b'),
                             ('scaleshift_1_w', 'scaleshift_1'),
                             ('scaleshift_1_b', 'scaleshift_1'),
                             ('scaleshift_1', 'scaleshift_1_data'),
                             ('scaleshift_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'scaleshift_1_w': {'shape': np.array([3]), 'value': np.array([1, 1, 1])},
                             'scaleshift_1_b': {'shape': np.array([3]), 'value': np.array([0, 0, 0])},
                             'scaleshift_1_data': {'shape': np.array([1, 227, 227, 3])}
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])}}
                                ,nodes_with_edges_only=True)

        graph.graph['layout'] = 'NHWC'
        convert_scale_shift_to_mul_add(graph)
        graph.clean_up()
        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1')
        self.assertTrue(flag, resp)

    # ScaleShift -> ScaleShift (can_be_fused=False)
    def test_scaleshift_can_be_fused(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'scaleshift_1'),
                             ('const_scaleshift_1_w', 'scaleshift_1_w'),
                             ('const_scaleshift_1_b', 'scaleshift_1_b'),
                             ('scaleshift_1_w', 'scaleshift_1'),
                             ('scaleshift_1_b', 'scaleshift_1'),
                             ('scaleshift_1', 'scaleshift_1_data'),
                             ('scaleshift_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'scaleshift_1_w': {'shape': np.array([3]), 'value': np.array([1, 1, 1])},
                             'scaleshift_1_b': {'shape': np.array([3]), 'value': np.array([0, 0, 0])},
                             'scaleshift_1': {'can_be_fused': False},
                             'scaleshift_1_data': {'shape': np.array([1, 227, 227, 3])}
                             })

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'scaleshift_1'),
                                 ('const_scaleshift_1_w', 'scaleshift_1_w'),
                                 ('const_scaleshift_1_b', 'scaleshift_1_b'),
                                 ('scaleshift_1_w', 'scaleshift_1'),
                                 ('scaleshift_1_b', 'scaleshift_1'),
                                 ('scaleshift_1', 'scaleshift_1_data'),
                                 ('scaleshift_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'const_scaleshift_1_w': {'shape': np.array([3]), 'value': np.array([1, 1, 1])},
                                 'scaleshift_1_w': {'shape': np.array([3]), 'value': np.array([1, 1, 1])},
                                 'const_scaleshift_1_b': {'shape': np.array([3]), 'value': np.array([0, 0, 0])},
                                 'scaleshift_1_b': {'shape': np.array([3]), 'value': np.array([0, 0, 0])},
                                 'scaleshift_1': {'can_be_fused': False},
                                 'scaleshift_1_data': {'shape': np.array([1, 227, 227, 3])}
                                 })

        convert_scale_shift_to_mul_add(graph)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'scaleshift_1_data')
        self.assertTrue(flag, resp)


class BatchNormDecomposition(unittest.TestCase):
    def test_bn_decomposition_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'bn_op'),
                             ('const_bn_const', 'bn_const'),
                             ('const_bn_beta', 'bn_beta'),
                             ('const_bn_mean', 'bn_mean'),
                             ('const_bn_var', 'bn_var'),
                             ('bn_const', 'bn_op'),
                             ('bn_beta', 'bn_op'),
                             ('bn_mean', 'bn_op'),
                             ('bn_var', 'bn_op'),
                             ('bn_op', 'bn_data'),
                             ('concat', 'concat_data'),
                             ('bn_data', 'concat'),
                             ('concat_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'bn_op': {'eps': 1.2},
                             'bn_const': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_beta': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_mean': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_var': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_data': {'shape': np.array([1, 227, 227, 3])},
                             'concat_data': {}
                             }, nodes_with_edges_only=True)

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
                                 ('mul_2_data', 'add_2'),
                                 ('const_add_2_w', 'add_2_w'),
                                 ('add_2_w', 'add_2'),
                                 ('add_2', 'add_2_data'),
                                 ('concat', 'concat_data'),
                                 ('add_2_data', 'concat'),
                                 ('concat_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'const_mul_1_w': {'shape': np.array([3]),
                                             'value': np.array([0.67419986, 0.55901699, 0.48795004])},
                                 'mul_1_w': {'shape': np.array([3]),
                                             'value': np.array([0.67419986, 0.55901699, 0.48795004])},
                                 'const_mul_2_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'mul_2_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'const_add_1_w': {'shape': np.array([3]),
                                             'value': np.array([-0.67419986, -1.11803399, -1.46385011])},
                                 'add_1_w': {'shape': np.array([3]),
                                             'value': np.array([-0.67419986, -1.11803399, -1.46385011])},
                                 'const_add_2_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'add_2_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'add_2_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1': {'can_be_fused': True},
                                 'mul_2': {'can_be_fused': True},
                                 'add_1': {'can_be_fused': True},
                                 'add_2': {'can_be_fused': True},
                                 'concat_data': {}
                                 }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NHWC'
        convert_batch_norm(graph)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_data')
        self.assertTrue(flag, resp)

    # 'can_be_fused': False for BatchNorm
    def test_bn_decomposition_2(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'bn_op'),
                             ('const_bn_const', 'bn_const'),
                             ('const_bn_beta', 'bn_beta'),
                             ('const_bn_mean', 'bn_mean'),
                             ('const_bn_var', 'bn_var'),
                             ('bn_const', 'bn_op'),
                             ('bn_beta', 'bn_op'),
                             ('bn_mean', 'bn_op'),
                             ('bn_var', 'bn_op'),
                             ('bn_op', 'bn_data'),
                             ('concat', 'concat_data'),
                             ('bn_data', 'concat'),
                             ('concat_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'bn_op': {'eps': 1.2, 'can_be_fused': False},
                             'bn_const': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_beta': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_mean': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_var': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_data': {'shape': np.array([1, 227, 227, 3])},
                             'concat_data': {}
                             })

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
                                 ('mul_2_data', 'add_2'),
                                 ('const_add_2_w', 'add_2_w'),
                                 ('add_2_w', 'add_2'),
                                 ('add_2', 'add_2_data'),
                                 ('concat', 'concat_data'),
                                 ('add_2_data', 'concat'),
                                 ('concat_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'const_mul_1_w': {'shape': np.array([3]),
                                             'value': np.array([0.67419986, 0.55901699, 0.48795004])},
                                 'mul_1_w': {'shape': np.array([3]),
                                             'value': np.array([0.67419986, 0.55901699, 0.48795004])},
                                 'const_mul_2_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'mul_2_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'const_add_1_w': {'shape': np.array([3]),
                                             'value': np.array([-0.67419986, -1.11803399, -1.46385011])},
                                 'add_1_w': {'shape': np.array([3]),
                                             'value': np.array([-0.67419986, -1.11803399, -1.46385011])},
                                 'const_add_2_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'add_2_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'add_2_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1': {'can_be_fused': False},
                                 'mul_2': {'can_be_fused': False},
                                 'add_1': {'can_be_fused': False},
                                 'add_2': {'can_be_fused': False},
                                 'concat_data': {}
                                 })

        graph.graph['layout'] = 'NHWC'
        convert_batch_norm(graph)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_data')
        self.assertTrue(flag, resp)

    # graph - NCHW
    # BatchNorm - NHWC
    def test_bn_decomposition_different_layouts_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'bn_op'),
                             ('const_bn_const', 'bn_const'),
                             ('const_bn_beta', 'bn_beta'),
                             ('const_bn_mean', 'bn_mean'),
                             ('const_bn_var', 'bn_var'),
                             ('bn_const', 'bn_op'),
                             ('bn_beta', 'bn_op'),
                             ('bn_mean', 'bn_op'),
                             ('bn_var', 'bn_op'),
                             ('bn_op', 'bn_data'),
                             ('concat', 'concat_data'),
                             ('bn_data', 'concat'),
                             ('concat_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'bn_op': {'eps': 1.2, 'data_format': 'NHWC'},
                             'bn_const': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_beta': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_mean': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_var': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_data': {'shape': np.array([1, 227, 227, 3])},
                             'concat_data': {}
                             }, nodes_with_edges_only=True)

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
                                 ('mul_2_data', 'add_2'),
                                 ('const_add_2_w', 'add_2_w'),
                                 ('add_2_w', 'add_2'),
                                 ('add_2', 'add_2_data'),
                                 ('concat', 'concat_data'),
                                 ('add_2_data', 'concat'),
                                 ('concat_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'const_mul_1_w': {'shape': np.array([3]),
                                             'value': np.array([0.67419986, 0.55901699, 0.48795004])},
                                 'mul_1_w': {'shape': np.array([3]),
                                             'value': np.array([0.67419986, 0.55901699, 0.48795004])},
                                 'const_mul_2_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'mul_2_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'const_add_1_w': {'shape': np.array([3]),
                                             'value': np.array([-0.67419986, -1.11803399, -1.46385011])},
                                 'add_1_w': {'shape': np.array([3]),
                                             'value': np.array([-0.67419986, -1.11803399, -1.46385011])},
                                 'const_add_2_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'add_2_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'add_2_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1': {'can_be_fused': True},
                                 'mul_2': {'can_be_fused': True},
                                 'add_1': {'can_be_fused': True},
                                 'add_2': {'can_be_fused': True},
                                 'concat_data': {}
                                 }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        convert_batch_norm(graph)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_data')
        self.assertTrue(flag, resp)

    # graph - NHWC
    # BatchNorm - NCHW
    def test_bn_decomposition_different_layouts_2(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'bn_op'),
                             ('const_bn_const', 'bn_const'),
                             ('const_bn_beta', 'bn_beta'),
                             ('const_bn_mean', 'bn_mean'),
                             ('const_bn_var', 'bn_var'),
                             ('bn_const', 'bn_op'),
                             ('bn_beta', 'bn_op'),
                             ('bn_mean', 'bn_op'),
                             ('bn_var', 'bn_op'),
                             ('bn_op', 'bn_data'),
                             ('concat', 'concat_data'),
                             ('bn_data', 'concat'),
                             ('concat_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 3, 227, 227])},
                             'bn_op': {'eps': 1.2, 'data_format': 'NCHW'},
                             'bn_const': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_beta': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_mean': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_var': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_data': {'shape': np.array([1, 3, 227, 227])},
                             'concat_data': {}
                             }, nodes_with_edges_only=True)

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
                                 ('mul_2_data', 'add_2'),
                                 ('const_add_2_w', 'add_2_w'),
                                 ('add_2_w', 'add_2'),
                                 ('add_2', 'add_2_data'),
                                 ('concat', 'concat_data'),
                                 ('add_2_data', 'concat'),
                                 ('concat_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 3, 227, 227])},
                                 'const_mul_1_w': {'shape': np.array([3, 1, 1]),
                                             'value': np.array([[[0.67419986]], [[0.55901699]], [[0.48795004]]])},
                                 'mul_1_w': {'shape': np.array([3, 1, 1]),
                                             'value': np.array([[[0.67419986]], [[0.55901699]], [[0.48795004]]])},
                                 'const_mul_2_w': {'shape': np.array([3, 1, 1]), 'value': np.array([[[1]], [[2]], [[3]]])},
                                 'mul_2_w': {'shape': np.array([3, 1, 1]), 'value': np.array([[[1]], [[2]], [[3]]])},
                                 'const_add_1_w': {'shape': np.array([3, 1, 1]),
                                             'value': np.array([[[-0.67419986]], [[-1.11803399]], [[-1.46385011]]])},
                                 'add_1_w': {'shape': np.array([3, 1, 1]),
                                             'value': np.array([[[-0.67419986]], [[-1.11803399]], [[-1.46385011]]])},
                                 'const_add_2_w': {'shape': np.array([3, 1, 1]), 'value': np.array([[[1]], [[2]], [[3]]])},
                                 'add_2_w': {'shape': np.array([3, 1, 1]), 'value': np.array([[[1]], [[2]], [[3]]])},
                                 'add_2_data': {'shape': np.array([1, 3, 227, 227])},
                                 'mul_1': {'can_be_fused': True},
                                 'mul_2': {'can_be_fused': True},
                                 'add_1': {'can_be_fused': True},
                                 'add_2': {'can_be_fused': True},
                                 'concat_data': {}
                                 }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NHWC'
        convert_batch_norm(graph)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_data')
        self.assertTrue(flag, resp)
