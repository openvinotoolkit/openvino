# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.back.FuseTransposesSequence import FuseTransposesSequence
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

# The dictionary with nodes attributes used to build various graphs. A key is the name of the node and the value is the
# dictionary with node attributes.
nodes_attributes = {
    'placeholder_1': {'name': 'placeholder_1', 'value': None, 'shape': None, 'type': 'Parameter', 'kind': 'op',
                      'op': 'Parameter'},
    'placeholder_1_data': {'name': 'placeholder_1_data', 'value': None, 'shape': None, 'kind': 'data',
                           'data_type': None},
    # Transpose layers
    'const_1': {'value': None, 'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'const_1_data': {'value': None, 'shape': None, 'kind': 'data'},

    'permute_1': {'type': 'Transpose', 'value': None, 'kind': 'op', 'op': 'Transpose'},
    'permute_1_data': {'value': None, 'shape': None, 'kind': 'data'},

    'const_2': {'value': None, 'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'const_2_data': {'value': None, 'shape': None, 'kind': 'data'},

    'permute_2': {'type': 'Transpose', 'value': None, 'kind': 'op', 'op': 'Transpose'},
    'permute_2_data': {'value': None, 'shape': None, 'kind': 'data'},

    'const_3': {'value': None, 'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'const_3_data': {'value': None, 'shape': None, 'kind': 'data'},

    'permute_3': {'type': 'Transpose', 'value': None, 'kind': 'op', 'op': 'Transpose'},
    'permute_3_data': {'value': None, 'shape': None, 'kind': 'data'},
    'op_output': {'op': 'Result', 'kind': 'op'}
}


class FuseTransposesSequenceTest(unittest.TestCase):
    def test_1(self):
        #
        #    NHWC           NCHW           NHWC
        #   Input->DATA->Transpose->DATA->Transpose->DATA  => Input->DATA
        #
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'permute_1'),
                             ('permute_1', 'permute_1_data'),
                             ('permute_1_data', 'permute_2'),
                             ('permute_2', 'permute_2_data'),
                             ('permute_2_data', 'op_output'),

                             ('const_1', 'const_1_data'),
                             ('const_1_data', 'permute_1', {'in': 1}),

                             ('const_2', 'const_2_data'),
                             ('const_2_data', 'permute_2', {'in': 1}),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},

                             'const_1_data': {'value': np.array([0, 3, 1, 2])},
                             'permute_1_data': {'shape': np.array([1, 3, 227, 227])},

                             'const_2_data': {'value': np.array([0, 2, 3, 1])},
                             'permute_2_data': {'shape': np.array([1, 227, 227, 3])},
                             }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])}},
                                nodes_with_edges_only=True)

        pattern = FuseTransposesSequence()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_2(self):
        #
        #   Input->DATA->Transpose->DATA->Transpose->DATA  => Input->DATA->Transpose->DATA
        #
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'permute_1'),
                             ('permute_1', 'permute_1_data'),
                             ('permute_1_data', 'permute_2'),
                             ('permute_2', 'permute_2_data'),
                             ('permute_2_data', 'op_output'),

                             ('const_1', 'const_1_data'),
                             ('const_1_data', 'permute_1', {'in': 1}),

                             ('const_2', 'const_2_data'),
                             ('const_2_data', 'permute_2', {'in': 1}),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'const_1': {'shape': np.array([4])},
                             'const_1_data': {'value': np.array([0, 3, 1, 2])},
                             'permute_1_data': {'shape': np.array([1, 3, 227, 227])},

                             'const_2': {'shape': np.array([4])},
                             'const_2_data': {'value': np.array([0, 1, 2, 3])},
                             'permute_2_data': {'shape': np.array([1, 3, 227, 227])},
                             }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'permute_1'),
                                 ('permute_1', 'permute_1_data'),
                                 ('permute_1_data', 'op_output'),

                                 ('const_1', 'const_1_data'),
                                 ('const_1_data', 'permute_1', {'in': 1}),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'const_1_data': {'value': np.array([0, 3, 1, 2])},
                                 'permute_1_data': {'shape': np.array([1, 3, 227, 227])},
                                 }, nodes_with_edges_only=True)

        pattern = FuseTransposesSequence()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)


if __name__ == '__main__':
    unittest.main()
