"""
 Copyright (C) 2018-2021 Intel Corporation

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

from mo.middle.passes.fusing.decomposition import convert_scale_shift_to_mul_add, convert_batch_norm
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph

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
    'bn_op': {'type': None, 'kind': 'op', 'op': 'BatchNormInference', 'can_be_fused': True},
    'const_bn_const': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Const'},
    'bn_const': {'value': None, 'shape': None, 'kind': 'data'},
    'const_bn_beta': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Const'},
    'bn_beta': {'value': None, 'shape': None, 'kind': 'data'},
    'const_bn_mean': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Const'},
    'bn_mean': {'value': None, 'shape': None, 'kind': 'data'},
    'const_bn_var': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Const'},
    'bn_var': {'value': None, 'shape': None, 'kind': 'data'},
    'bn_data': {'value': None, 'shape': None, 'kind': 'data'},
    'var_add_eps': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Add'},
    'var_add_eps_data': {'value': None, 'shape': None, 'kind': 'data'},
    'neg_mul_mean': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Mul'},
    'neg_mul_mean_data': {'value': None, 'shape': None, 'kind': 'data'},
    'shift': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Mul'},
    'shift_data': {'value': None, 'shape': None, 'kind': 'data'},
    'const_bn_eps': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Const'},
    'bn_eps': {'value': None, 'shape': None, 'kind': 'data'},
    'const_bn_neg': {'value': np.float32(-1.), 'kind': 'op', 'op': 'Const'},
    'bn_neg': {'value': np.float32(-1.), 'kind': 'data'},
    'const_bn_pow_val': {'value': np.float32(-0.5), 'kind': 'op', 'op': 'Const'},
    'bn_pow_val': {'value': np.float32(-0.5), 'kind': 'data'},
    'bn_pow': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Pow'},
    'bn_pow_data': {'value': None, 'shape': None, 'kind': 'data'},
    'non_const_bn_mean': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Const'},
    'non_const_bn_var': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Const'},
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
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])}},
                                nodes_with_edges_only=True)

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

    def test_bn_non_constant_var_mean(self):
        # TODO: Rewrite unit test and add unit tests for NCHW
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'bn_op'),
                             ('const_bn_const', 'bn_const'),
                             ('const_bn_beta', 'bn_beta'),
                             ('non_const_bn_mean', 'bn_mean'),
                             ('non_const_bn_var', 'bn_var'),
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
                             'bn_op': {'eps': np.float32(1.2), 'can_be_fused':False},
                             'bn_const': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_beta': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_mean': {'shape': np.array([3]), 'value': None},
                             'bn_var': {'shape': np.array([3]), 'value': None},
                             'bn_data': {'shape': np.array([1, 227, 227, 3])},
                             'concat_data': {}
                             }, nodes_with_edges_only=True)
        graph.graph['layout'] = 'NHWC'
        convert_batch_norm(graph)
        graph.clean_up()

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('non_const_bn_var', 'bn_var'),
                                 ('bn_var', 'var_add_eps'),
                                 ('const_bn_eps', 'bn_eps'),
                                 ('bn_eps', 'var_add_eps'),
                                 ('var_add_eps', 'var_add_eps_data'),
                                 ('var_add_eps_data', 'bn_pow'),
                                 ('const_bn_pow_val', 'bn_pow_val'),
                                 ('bn_pow_val', 'bn_pow'),
                                 ('bn_pow', 'bn_pow_data'),
                                 ('non_const_bn_mean', 'bn_mean'),
                                 ('bn_mean', 'neg_mul_mean'),
                                 ('const_bn_neg', 'bn_neg'),
                                 ('bn_neg', 'neg_mul_mean'),
                                 ('neg_mul_mean', 'neg_mul_mean_data'),
                                 ('bn_pow_data', 'shift'),
                                 ('neg_mul_mean_data', 'shift'),
                                 ('shift', 'shift_data'),
                                 ('bn_pow_data', 'mul_1'),
                                 ('mul_1', 'mul_1_data'),
                                 ('mul_1_data', 'add_1'),
                                 ('shift_data', 'add_1'),
                                 ('add_1', 'add_1_data'),
                                 ('add_1_data', 'mul_2'),
                                 ('const_bn_const', 'bn_const'),
                                 ('bn_const', 'mul_2'),
                                 ('mul_2', 'mul_2_data'),
                                 ('mul_2_data', 'add_2'),
                                 ('const_bn_beta', 'bn_beta'),
                                 ('bn_beta', 'add_2'),
                                 ('add_2', 'add_2_data'),
                                 ('concat', 'concat_data'),
                                 ('add_2_data', 'concat'),
                                 ('concat_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'bn_eps': {'value': np.float32(1.2)},
                                 'bn_var': {'shape':np.array([3])},
                                 'bn_mean': {'shape':np.array([3])},
                                 'bn_const': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'bn_beta': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'mul_1': {'can_be_fused': False},
                                 'mul_2': {'can_be_fused': False},
                                 'add_1': {'can_be_fused': False},
                                 'add_2': {'can_be_fused': False},
                                 'concat_data': {}
                                 })
        (flag, resp) = compare_graphs(graph, graph_ref, 'op_output')
        self.assertTrue(flag, resp)


