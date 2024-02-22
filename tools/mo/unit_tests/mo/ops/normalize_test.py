# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.common.partial_infer.elemental import copy_shape_infer
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph

nodes_attributes = {'node_1': {'type': 'Identity', 'kind': 'op'},
                    'norm': {'type': 'Normalize', 'kind': 'op'},
                    'node_3': {'type': 'Identity', 'kind': 'op'},
                    'op_output': { 'kind': 'op', 'op': 'Result'}
                    }


class TestNormalize(unittest.TestCase):
    def test_region_infer(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'norm'),
                             ('norm', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227]), 'value': None},
                             'norm': {}
                             })

        norm_node = Node(graph, 'norm')
        copy_shape_infer(norm_node)
        exp_shape = np.array([1, 3, 227, 227])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])
