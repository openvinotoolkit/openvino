# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from extensions.ops.spatial_transformer import SpatialTransformOp
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'node_1': {'type': 'Identity', 'kind': 'op'},
                    'node_2': {'type': 'Identity', 'kind': 'op'},
                    'st': {'type': 'SpatialTransform', 'kind': 'op'},
                    'node_3': {'type': 'Identity', 'kind': 'op'},
                    'op_output': { 'kind': 'op', 'op': 'Result'}
                    }


class TestSpatialTransformInfer(unittest.TestCase):
    def test_sp_transform_concat_infer(self):
        graph = build_graph(nodes_attributes,
                            [
                                ('node_1', 'st'),
                                ('node_2', 'st'),
                                ('st', 'node_3'),
                                ('node_3', 'op_output')
                            ],
                            {
                                'node_3': {'shape': None},
                                'node_1': {'shape': np.array([1, 3, 227, 227])},
                                'node_2': {'shape': np.array([1, 3, 227, 227])},
                                'st': {}
                            })

        st_node = Node(graph, 'st')
        SpatialTransformOp.sp_infer(st_node)
        exp_shape = np.array([1, 3, 227, 227])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_sp_transform_with_output_params_infer(self):
        graph = build_graph(nodes_attributes,
                            [
                                ('node_1', 'st'),
                                ('node_2', 'st'),
                                ('st', 'node_3'),
                                ('node_3', 'op_output')
                            ],
                            {
                                'node_3': {'shape': None},
                                'node_1': {'shape': np.array([1, 3, 227, 227])},
                                'node_2': {'shape': np.array([1, 3, 227, 227])},
                                'st': {'output_H': 200, 'output_W': 15}
                            })

        st_node = Node(graph, 'st')
        SpatialTransformOp.sp_infer(st_node)
        exp_shape = np.array([1, 3, 200, 15])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])
