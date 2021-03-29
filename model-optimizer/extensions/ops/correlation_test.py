# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from extensions.ops.correlation import CorrelationOp
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'node_1': {'type': 'Identity', 'kind': 'op'},
                    'node_2': {'type': 'Identity', 'kind': 'op'},
                    'corr': {'type': 'Correlation', 'kind': 'op'},
                    'node_3': {'type': 'Identity', 'kind': 'op'},
                    'op_output': {'kind': 'op', 'op': 'Result'}
                    }


class TestConcatPartialInfer(unittest.TestCase):
    def test_tf_concat_infer(self):
        graph = build_graph(nodes_attributes,
                            [
                                ('node_1', 'corr'),
                                ('node_2', 'corr'),
                                ('corr', 'node_3'),
                                ('node_3', 'op_output')
                            ],
                            {
                                'node_3': {'shape': None},
                                'node_1': {'shape': np.array([1, 3, 227, 227])},
                                'node_2': {'shape': np.array([1, 3, 227, 227])},
                                'corr': {'pad': 20,
                                         'kernel_size': 1,
                                         'max_displacement': 20,
                                         'stride_1': 1,
                                         'stride_2': 2,
                                         'single_direction': 0,
                                         'do_abs': False,
                                         'correlation_type': 0}
                            })

        corr_node = Node(graph, 'corr')
        CorrelationOp.corr_infer(corr_node)
        exp_shape = np.array([1, 441, 227, 227])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])
