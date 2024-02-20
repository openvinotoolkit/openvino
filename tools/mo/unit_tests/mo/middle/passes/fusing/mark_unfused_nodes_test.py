# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.middle.passes.fusing.mark_unfused_nodes import mark_unfused_nodes
from unit_tests.utils.graph import build_graph

nodes_attributes = {
    'placeholder_1': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # ScaleShift layer
    'scaleshift_1': {'type': 'ScaleShift', 'kind': 'op', 'op': 'ScaleShift'},
    'scaleshift_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'scaleshift_1_b': {'value': None, 'shape': None, 'kind': 'data'},
    'scaleshift_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Mul and Add operations
    'mul_1': {'type': 'Mul', 'kind': 'op', 'op': 'Mul'},
    'mul_1_w': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'mul_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'add_1': {'type': 'Add', 'kind': 'op', 'op': 'Add'},
    'add_1_w': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'add_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # Mul2 and Add2 operations
    'mul_2': {'type': 'Mul', 'kind': 'op', 'op': 'Mul'},
    'mul_2_w': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'mul_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'add_2': {'type': 'Add', 'kind': 'op', 'op': 'Add'},
    'add_2_w': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'add_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # Concat1 operation
    'concat_1': {'type': 'Concat', 'kind': 'op', 'op': 'Concat'},
    'concat_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Convolutions
    'conv_1': {'type': 'Convolution', 'kind': 'op', 'op': 'Conv2D', 'layout': 'NHWC'},
    'conv_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_1_b': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_2': {'type': 'Convolution', 'kind': 'op', 'op': 'Conv2D', 'layout': 'NHWC'},
    'conv_2_w': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_2_b': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_2_data': {'value': None, 'shape': None, 'kind': 'data'},
    # FullyConnected
    'fc_1': {'type': 'MatMul', 'kind': 'op', 'layout': 'NHWC'},
    'fc_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'fc_1_b': {'value': None, 'shape': None, 'kind': 'data'},
    'fc_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Placeholders
    'placeholder_2': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_3': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_3_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'op_output': { 'kind': 'op', 'op': 'Result'}
}


# Unit tests for forward and backward bfs (forward_bfs, backward_bfs)
class MarkFusedNodes(unittest.TestCase):
    def test_mark_unfused_nodes_1(self):
        # Placeholder->ScaleShift->Mul->Add
        graph = build_graph(nodes_attributes,
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
                             })

        graph.graph['layout'] = 'NHWC'

        mark_unfused_nodes(graph, '.*mul.*')

        self.assertFalse(graph.node['mul_1']['can_be_fused'], "can_be_fused should be False")
        self.assertFalse(graph.node['mul_2']['can_be_fused'], "can_be_fused should be False")
        self.assertTrue(graph.node['add_1']['can_be_fused'], "can_be_fused should be True")

    def test_mark_unfused_nodes_2(self):
        # Placeholder->ScaleShift->Mul->Add
        graph = build_graph(nodes_attributes,
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
                             })
        graph.graph['layout'] = 'NHWC'

        mark_unfused_nodes(graph, '.*')

        self.assertFalse(graph.node['mul_1']['can_be_fused'], "can_be_fused should be False")
        self.assertFalse(graph.node['mul_2']['can_be_fused'], "can_be_fused should be False")
        self.assertFalse(graph.node['add_1']['can_be_fused'], "can_be_fused should be False")
        self.assertFalse(graph.node['placeholder_1']['can_be_fused'], "can_be_fused should be False")
        self.assertFalse(graph.node['concat_1']['can_be_fused'], "can_be_fused should be False")

    def test_mark_unfused_nodes_3(self):
        # Placeholder->ScaleShift->Mul->Add
        graph = build_graph(nodes_attributes,
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
                             })
        graph.graph['layout'] = 'NHWC'

        mark_unfused_nodes(graph, 'mul_1,add_1')

        self.assertFalse(graph.node['mul_1']['can_be_fused'], "can_be_fused should be False")
        self.assertFalse(graph.node['add_1']['can_be_fused'], "can_be_fused should be False")
        self.assertTrue(graph.node['mul_2']['can_be_fused'], "can_be_fused should be True")

    def test_mark_unfused_nodes_4(self):
        # Placeholder->ScaleShift->Mul->Add
        graph = build_graph(nodes_attributes,
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
                             ('mul_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('placeholder_1_data', 'concat_1'),
                             ('concat_1_data', 'op_output')

                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_2_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'add_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'mul_2_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             })
        graph.graph['layout'] = 'NHWC'

        mark_unfused_nodes(graph, '')

        self.assertTrue(graph.node['mul_1']['can_be_fused'], "can_be_fused should be True")
        self.assertTrue(graph.node['add_1']['can_be_fused'], "can_be_fused should be True")
        self.assertTrue(graph.node['mul_2']['can_be_fused'], "can_be_fused should be True")

    def test_mark_unfused_nodes_5(self):
        # Placeholder->ScaleShift->Mul->Add
        graph = build_graph(nodes_attributes,
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
                             ('mul_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('placeholder_1_data', 'concat_1'),
                             ('concat_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_2_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'add_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'mul_2_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             })
        graph.graph['layout'] = 'NCHW'

        mark_unfused_nodes(graph, '')

        self.assertTrue(graph.node['mul_1']['can_be_fused'], "can_be_fused should be True")
        self.assertTrue(graph.node['add_1']['can_be_fused'], "can_be_fused should be True")
        self.assertTrue(graph.node['mul_2']['can_be_fused'], "can_be_fused should be True")

        def test_mark_unfused_nodes_5(self):
            # Placeholder->ScaleShift->Mul->Add
            graph = build_graph(nodes_attributes,
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
                                 ('mul_2_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('placeholder_1_data', 'concat_1'),
                                 ('concat_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_2_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'add_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'mul_2_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 })
            graph.graph['layout'] = 'NCHW'

            mark_unfused_nodes(graph, '')

            self.assertFalse(graph.node['mul_1']['can_be_fused'], "can_be_fused should be False")
            self.assertFalse(graph.node['add_1']['can_be_fused'], "can_be_fused should be False")
            self.assertFalse(graph.node['mul_2']['can_be_fused'], "can_be_fused should be False")

    def test_mark_unfused_nodes_6(self):
        # Placeholder->ScaleShift->Mul->Add
        graph = build_graph(nodes_attributes,
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
                             ('mul_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('placeholder_1_data', 'concat_1'),
                             ('concat_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_2_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'add_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'mul_2_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             })
        graph.graph['layout'] = 'NHWC'

        mark_unfused_nodes(graph, '')

        self.assertTrue(graph.node['mul_1']['can_be_fused'], "can_be_fused should be True")
        self.assertTrue(graph.node['add_1']['can_be_fused'], "can_be_fused should be True")
        self.assertTrue(graph.node['mul_2']['can_be_fused'], "can_be_fused should be True")
