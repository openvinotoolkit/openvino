# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.common.partial_infer.elemental import copy_shape_infer
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph

nodes_attributes = {'node_1': {'type': 'Identity', 'kind': 'op'},
                    'grn': {'type': 'GRN', 'kind': 'op'},
                    'node_3': {'type': 'Identity', 'kind': 'op'},
                    'op_output': {'kind': 'op', 'op': 'Result'},
                    }


class TestGRNOp(unittest.TestCase):
    def test_grn_infer(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'grn'),
                             ('grn', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227])},
                             'grn': {'bias': 1}
                             })

        grn_node = Node(graph, 'grn')
        copy_shape_infer(grn_node)
        exp_shape = np.array([1, 3, 227, 227])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])
