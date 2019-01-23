"""
 Copyright (c) 2018 Intel Corporation

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

from mo.middle.passes.eliminate import graph_clean_up
from mo.middle.passes.fusing.decomposition import convert_scale_shift_to_mul_add, convert_batch_norm, \
    convert_bn_to_mul_add
from mo.utils.unittest.graph import build_graph, compare_graphs

nodes_attributes = {
    'placeholder_1': {'shape': None, 'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_2': {'shape': None, 'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
    'placeholder_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # ScaleShift layer
    'scaleshift_1': {'type': 'ScaleShift', 'kind': 'op', 'op': 'ScaleShift', 'axis': 0},
    'scaleshift_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'scaleshift_1_b': {'value': None, 'shape': None, 'kind': 'data'},
    'scaleshift_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Mul and Add operations
    'mul_1': {'type': None, 'value': None, 'kind': 'op', 'op': 'Mul'},
    'mul_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'mul_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'add_1': {'type': None, 'kind': 'op', 'op': 'Add'},
    'add_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'add_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Mul and Add operations
    'mul_2': {'type': None, 'kind': 'op', 'op': 'Mul'},
    'mul_2_w': {'value': None, 'shape': None, 'kind': 'data'},
    'mul_2_data': {'value': None, 'shape': None, 'kind': 'data'},
    'add_2': {'type': None, 'kind': 'op', 'op': 'Add'},
    'add_2_w': {'value': None, 'shape': None, 'kind': 'data'},
    'add_2_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Reshape
    'placeholder_2/Reshape_': {'type': 'Reshape', 'kind': 'op', 'op': 'Reshape'},
    'placeholder_2/Reshape_data': {'value': None, 'shape': None, 'kind': 'data'},
    # BatchNorm operation
    'bn_op': {'type': None, 'kind': 'op', 'op': 'BatchNorm', 'can_be_fused': True},
    'bn_const': {'value': None, 'shape': None, 'kind': 'data'},
    'bn_beta': {'value': None, 'shape': None, 'kind': 'data'},
    'bn_mean': {'value': None, 'shape': None, 'kind': 'data'},
    'bn_var': {'value': None, 'shape': None, 'kind': 'data'},
    'bn_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Concat1 operation
    'concat': {'type': 'Concat', 'kind': 'op', 'op': 'Concat'},
    'concat_data': {'value': None, 'shape': None, 'kind': 'data'},
}


class ScaleShiftToMulAdd(unittest.TestCase):
    # ScaleShift -> Mul
    def test_scaleshift_to_mul_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'scaleshift_1'),
                             ('scaleshift_1_w', 'scaleshift_1'),
                             ('scaleshift_1', 'scaleshift_1_data'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'scaleshift_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'scaleshift_1_data': {'is_output': True}
                             })

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'scaleshift_1_data'),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'mul_1': {'can_be_fused': True},
                                 'scaleshift_1_data': {'is_output': True}
                                 })

        graph.graph['layout'] = 'NHWC'
        convert_scale_shift_to_mul_add(graph)
        graph_clean_up(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'scaleshift_1_data')
        self.assertTrue(flag, resp)

    # ScaleShift  2 inputs-> Mul
    def test_scaleshift2_to_mul(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_2', 'placeholder_2_data'),
                             ('placeholder_1_data', 'scaleshift_1'),
                             ('placeholder_2_data', 'scaleshift_1'),
                             ('scaleshift_1', 'scaleshift_1_data'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'placeholder_2_data': {'shape': np.array([1, 227])},
                             'scaleshift_1_data': {'is_output': True}
                             })

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_2', 'placeholder_2_data'),
                                 ('placeholder_2_data', 'placeholder_2/Reshape_'),
                                 ('placeholder_2/Reshape_', 'placeholder_2/Reshape_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('placeholder_2/Reshape_data', 'mul_1'),
                                 ('mul_1', 'scaleshift_1_data'),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'placeholder_2_data': {'shape': np.array([1, 227])},
                                 'placeholder_2/Reshape_': {'dim': np.array([1, 227, 1, 1])},
                                 'placeholder_2/Reshape_data': {'shape': np.array([1, 227, 1, 1])},
                                 'mul_1': {'can_be_fused': True},
                                 'scaleshift_1_data': {'is_output': True}
                                 })

        graph.graph['layout'] = 'NHWC'
        convert_scale_shift_to_mul_add(graph)
        graph_clean_up(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'scaleshift_1_data')
        self.assertTrue(flag, resp)

    # ScaleShift  2 inputs-> Mul (axis = 1)
    def test_scaleshift2_axis1_to_mul(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_2', 'placeholder_2_data'),
                             ('placeholder_1_data', 'scaleshift_1'),
                             ('placeholder_2_data', 'scaleshift_1'),
                             ('scaleshift_1', 'scaleshift_1_data'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'placeholder_2_data': {'shape': np.array([227])},
                             'scaleshift_1': {'axis': 1},
                             'scaleshift_1_data': {'is_output': True}
                             })

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_2', 'placeholder_2_data'),
                                 ('placeholder_2_data', 'placeholder_2/Reshape_'),
                                 ('placeholder_2/Reshape_', 'placeholder_2/Reshape_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('placeholder_2/Reshape_data', 'mul_1'),
                                 ('mul_1', 'scaleshift_1_data'),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'placeholder_2_data': {'shape': np.array([227])},
                                 'placeholder_2/Reshape_': {'dim': np.array([1, 227, 1, 1])},
                                 'placeholder_2/Reshape_data': {'shape': np.array([1, 227, 1, 1])},
                                 'mul_1': {'can_be_fused': True},
                                 'scaleshift_1_data': {'is_output': True}
                                 })

        graph.graph['layout'] = 'NHWC'
        convert_scale_shift_to_mul_add(graph)
        graph_clean_up(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'scaleshift_1_data')
        self.assertTrue(flag, resp)


    # ScaleShift -> Mul (Zero biases)
    def test_scaleshift_to_mul_2(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'scaleshift_1'),
                             ('scaleshift_1_w', 'scaleshift_1'),
                             ('scaleshift_1_b', 'scaleshift_1'),
                             ('scaleshift_1', 'scaleshift_1_data'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'scaleshift_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'scaleshift_1_b': {'shape': np.array([3]), 'value': np.array([0, 0, 0])},
                             'scaleshift_1_data': {'is_output': True}
                             })

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'scaleshift_1_data'),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'mul_1': {'can_be_fused': True},
                                 'scaleshift_1_data': {'is_output': True}
                                 })

        graph.graph['layout'] = 'NHWC'
        convert_scale_shift_to_mul_add(graph)
        graph_clean_up(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'scaleshift_1_data')
        self.assertTrue(flag, resp)

    # ScaleShift -> Mul->Add
    def test_scaleshift_to_mul_add(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'scaleshift_1'),
                             ('scaleshift_1_w', 'scaleshift_1'),
                             ('scaleshift_1_b', 'scaleshift_1'),
                             ('scaleshift_1', 'scaleshift_1_data'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'scaleshift_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'scaleshift_1_b': {'shape': np.array([3]), 'value': np.array([3, 2, 1])},
                             'scaleshift_1_data': {'is_output': True}
                             })

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'mul_1_data'),
                                 ('mul_1_data', 'add_1'),
                                 ('add_1_w', 'add_1'),
                                 ('add_1', 'scaleshift_1_data'),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'add_1_w': {'shape': np.array([3]), 'value': np.array([3, 2, 1])},
                                 'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'add_1': {'can_be_fused': True},
                                 'mul_1': {'can_be_fused': True},
                                 'scaleshift_1_data': {'is_output': True}
                                 })

        graph.graph['layout'] = 'NHWC'
        convert_scale_shift_to_mul_add(graph)
        graph_clean_up(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'scaleshift_1_data')
        self.assertTrue(flag, resp)

    # ScaleShift -> None (Zero weights and biases)
    def test_scaleshift_to_nothing(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'scaleshift_1'),
                             ('scaleshift_1_w', 'scaleshift_1'),
                             ('scaleshift_1_b', 'scaleshift_1'),
                             ('scaleshift_1', 'scaleshift_1_data'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'scaleshift_1_w': {'shape': np.array([3]), 'value': np.array([1, 1, 1])},
                             'scaleshift_1_b': {'shape': np.array([3]), 'value': np.array([0, 0, 0])},
                             'scaleshift_1_data': {'shape': np.array([1, 227, 227, 3]), 'is_output': True}
                             })

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data')],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3]), 'is_output': True}})

        graph.graph['layout'] = 'NHWC'
        convert_scale_shift_to_mul_add(graph)
        graph_clean_up(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1_data')
        self.assertTrue(flag, resp)

    # ScaleShift -> ScaleShift (can_be_fused=False)
    def test_scaleshift_can_be_fused(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'scaleshift_1'),
                             ('scaleshift_1_w', 'scaleshift_1'),
                             ('scaleshift_1_b', 'scaleshift_1'),
                             ('scaleshift_1', 'scaleshift_1_data'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'scaleshift_1_w': {'shape': np.array([3]), 'value': np.array([1, 1, 1])},
                             'scaleshift_1_b': {'shape': np.array([3]), 'value': np.array([0, 0, 0])},
                             'scaleshift_1': {'can_be_fused': False},
                             'scaleshift_1_data': {'shape': np.array([1, 227, 227, 3]), 'is_output': True}
                             })

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'scaleshift_1'),
                                 ('scaleshift_1_w', 'scaleshift_1'),
                                 ('scaleshift_1_b', 'scaleshift_1'),
                                 ('scaleshift_1', 'scaleshift_1_data'),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'scaleshift_1_w': {'shape': np.array([3]), 'value': np.array([1, 1, 1])},
                                 'scaleshift_1_b': {'shape': np.array([3]), 'value': np.array([0, 0, 0])},
                                 'scaleshift_1': {'can_be_fused': False},
                                 'scaleshift_1_data': {'shape': np.array([1, 227, 227, 3]), 'is_output': True}
                                 })

        convert_scale_shift_to_mul_add(graph)
        graph_clean_up(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'scaleshift_1_data')
        self.assertTrue(flag, resp)


class BatchNormDecomposition(unittest.TestCase):
    def test_bn_decomposition_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'bn_op'),
                             ('bn_const', 'bn_op'),
                             ('bn_beta', 'bn_op'),
                             ('bn_mean', 'bn_op'),
                             ('bn_var', 'bn_op'),
                             ('bn_op', 'bn_data'),
                             ('concat', 'concat_data'),
                             ('bn_data', 'concat')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'bn_op': {'eps': 1.2},
                             'bn_const': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_beta': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_mean': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_var': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_data': {'shape': np.array([1, 227, 227, 3])},
                             'concat_data': {'is_output': True}
                             })

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'mul_1_data'),
                                 ('mul_1_data', 'add_1'),
                                 ('add_1_w', 'add_1'),
                                 ('add_1', 'add_1_data'),
                                 ('add_1_data', 'mul_2'),
                                 ('mul_2_w', 'mul_2'),
                                 ('mul_2', 'mul_2_data'),
                                 ('mul_2_data', 'add_2'),
                                 ('add_2_w', 'add_2'),
                                 ('add_2', 'add_2_data'),
                                 ('concat', 'concat_data'),
                                 ('add_2_data', 'concat')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_w': {'shape': np.array([3]),
                                             'value': np.array([0.67419986, 0.55901699, 0.48795004])},
                                 'mul_2_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'add_1_w': {'shape': np.array([3]),
                                             'value': np.array([-0.67419986, -1.11803399, -1.46385011])},
                                 'add_2_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'add_2_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1': {'can_be_fused': True},
                                 'mul_2': {'can_be_fused': True},
                                 'add_1': {'can_be_fused': True},
                                 'add_2': {'can_be_fused': True},
                                 'concat_data': {'is_output': True}
                                 })

        graph.graph['layout'] = 'NHWC'
        convert_batch_norm(graph)
        graph_clean_up(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_data')
        self.assertTrue(flag, resp)

    # 'can_be_fused': False for BatchNorm
    def test_bn_decomposition_2(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'bn_op'),
                             ('bn_const', 'bn_op'),
                             ('bn_beta', 'bn_op'),
                             ('bn_mean', 'bn_op'),
                             ('bn_var', 'bn_op'),
                             ('bn_op', 'bn_data'),
                             ('concat', 'concat_data'),
                             ('bn_data', 'concat')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'bn_op': {'eps': 1.2, 'can_be_fused': False},
                             'bn_const': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_beta': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_mean': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_var': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_data': {'shape': np.array([1, 227, 227, 3])},
                             'concat_data': {'is_output': True}
                             })

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'mul_1_data'),
                                 ('mul_1_data', 'add_1'),
                                 ('add_1_w', 'add_1'),
                                 ('add_1', 'add_1_data'),
                                 ('add_1_data', 'mul_2'),
                                 ('mul_2_w', 'mul_2'),
                                 ('mul_2', 'mul_2_data'),
                                 ('mul_2_data', 'add_2'),
                                 ('add_2_w', 'add_2'),
                                 ('add_2', 'add_2_data'),
                                 ('concat', 'concat_data'),
                                 ('add_2_data', 'concat')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_w': {'shape': np.array([3]),
                                             'value': np.array([0.67419986, 0.55901699, 0.48795004])},
                                 'mul_2_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'add_1_w': {'shape': np.array([3]),
                                             'value': np.array([-0.67419986, -1.11803399, -1.46385011])},
                                 'add_2_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'add_2_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1': {'can_be_fused': False},
                                 'mul_2': {'can_be_fused': False},
                                 'add_1': {'can_be_fused': False},
                                 'add_2': {'can_be_fused': False},
                                 'concat_data': {'is_output': True}
                                 })

        graph.graph['layout'] = 'NHWC'
        convert_batch_norm(graph)
        graph_clean_up(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_data')
        self.assertTrue(flag, resp)

    def test_caffe_bn_decomposition_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'bn_op'),
                             ('bn_mean', 'bn_op'),
                             ('bn_var', 'bn_op'),
                             ('bn_op', 'bn_data'),
                             ('concat', 'concat_data'),
                             ('bn_data', 'concat')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'bn_op': {'epsilon': 1.2, 'op': 'BatchNormalization'},
                             'bn_mean': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_var': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_data': {'shape': np.array([1, 227, 227, 3])},
                             'concat_data': {'is_output': True}
                             })

        del graph['placeholder_1']['placeholder_1_data'][0]['in']
        del graph['bn_op']['bn_data'][0]['in']

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'mul_1_data'),
                                 ('mul_1_data', 'add_1'),
                                 ('add_1_w', 'add_1'),
                                 ('add_1', 'add_1_data'),
                                 ('concat', 'concat_data'),
                                 ('add_1_data', 'concat')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_w': {'shape': np.array([3]),
                                             'value': np.array([0.67419986, 0.55901699, 0.48795004])},
                                 'add_1_w': {'shape': np.array([3]),
                                             'value': np.array([-0.67419986, -1.11803399, -1.46385011])},
                                 'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1': {'can_be_fused': True},
                                 'add_1': {'can_be_fused': True},
                                 'concat_data': {'is_output': True}
                                 })

        graph.graph['layout'] = 'NHWC'
        convert_bn_to_mul_add(graph)
        graph_clean_up(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_data')
        self.assertTrue(flag, resp)

    # 'can_be_fused': False for BatchNormalization
    def test_caffe_bn_decomposition_2(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'bn_op'),
                             ('bn_mean', 'bn_op'),
                             ('bn_var', 'bn_op'),
                             ('bn_op', 'bn_data'),
                             ('concat', 'concat_data'),
                             ('bn_data', 'concat')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'bn_op': {'epsilon': 1.2, 'op': 'BatchNormalization', 'can_be_fused': False},
                             'bn_mean': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_var': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'bn_data': {'shape': np.array([1, 227, 227, 3])},
                             'concat_data': {'is_output': True}
                             })

        del graph['placeholder_1']['placeholder_1_data'][0]['in']
        del graph['bn_op']['bn_data'][0]['in']

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'mul_1_data'),
                                 ('mul_1_data', 'add_1'),
                                 ('add_1_w', 'add_1'),
                                 ('add_1', 'add_1_data'),
                                 ('concat', 'concat_data'),
                                 ('add_1_data', 'concat')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_w': {'shape': np.array([3]),
                                             'value': np.array([0.67419986, 0.55901699, 0.48795004])},
                                 'add_1_w': {'shape': np.array([3]),
                                             'value': np.array([-0.67419986, -1.11803399, -1.46385011])},
                                 'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1': {'can_be_fused': False},
                                 'add_1': {'can_be_fused': False},
                                 'concat_data': {'is_output': True}
                                 })

        graph.graph['layout'] = 'NHWC'
        convert_bn_to_mul_add(graph)
        graph_clean_up(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_data')
        self.assertTrue(flag, resp)
